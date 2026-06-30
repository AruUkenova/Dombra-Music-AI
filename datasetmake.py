import os
import random
import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm

# --- Настройки воспроизводимости (Seed) ---
SEED = 67
random.seed(SEED)
np.random.seed(SEED)

# --- Настройки ---
SR = 22050
TRACK_DURATION = 10.0
NUM_TRACKS = 2000

MAX_FRET = 19
MAX_STRETCH = 4
LOWER_BASE_MIDI = 55
UPPER_BASE_MIDI = 50

SOURCE_FOLDER = 'clean_samples'
OUTPUT_FOLDER = 'generated_dataset'
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Новое: мы будем сохранять метки с частотой кадров (frame_rate)
FRAME_RATE = SR / 512.0  # 43.0664 Hz
SILENCE_CLASS = 20 # 21-й класс для тишины



def load_source_notes(folder_path):
    notes = {'lower': {}, 'upper': {}}
    for subdir in sorted(os.listdir(folder_path)):
        subdir_path = os.path.join(folder_path, subdir)
        if not os.path.isdir(subdir_path):
            continue
            
        string_type = None
        fret_str = ""
        
        if subdir.startswith('n'):
            string_type = 'lower'
            fret_str = subdir[1:]
        elif subdir.startswith('v'):
            string_type = 'upper'
            fret_str = subdir[1:]
            
        if not string_type or not fret_str:
            continue
            
        try:
            fret_num = int(fret_str)
        except ValueError:
            continue
            
        if fret_num not in notes[string_type]:
            notes[string_type][fret_num] = []
            
        for filename in sorted(os.listdir(subdir_path)):
            if not filename.endswith('.wav'):
                continue
            filepath = os.path.join(subdir_path, filename)
            try:
                audio, _ = librosa.load(filepath, sr=SR)
                notes[string_type][fret_num].append(audio)
            except Exception as e:
                print(f"Ошибка при загрузке {filepath}: {e}")
                
    return notes

def get_realistic_fret_pair(avail_lower, avail_upper):
    # Упрощаем стили для более качественного начального обучения
    # Стиль 0: Одиночная нота на нижней струне (верхняя молчит)
    # Стиль 1: Одиночная нота на верхней струне (нижняя молчит)
    # Стиль 2: Простой аккорд (обе струны)
    # Стиль 3: Тишина (пауза)
    
    style = random.choices([0, 1, 2, 3], weights=[0.35, 0.35, 0.20, 0.10])[0]

    if style == 0:
        f_l = random.choice(avail_lower)
        return f_l, SILENCE_CLASS

    elif style == 1:
        f_u = random.choice(avail_upper)
        return SILENCE_CLASS, f_u

    elif style == 2:
        # Простой аккорд с небольшой растяжкой
        f_l = random.choice(avail_lower)
        valid_upper = [f for f in avail_upper if f == 0 or abs(f - f_l) <= MAX_STRETCH]
        f_u = random.choice(valid_upper) if valid_upper else 0
        return f_l, f_u

    elif style == 3:
        # Пауза
        return SILENCE_CLASS, SILENCE_CLASS

    return SILENCE_CLASS, SILENCE_CLASS


def soft_clip(audio):
    """
    ИСПРАВЛЕНИЕ 3: Двухэтапная обработка.
    Шаг 1 — нормализация до уровня, при котором tanh не насыщается.
    Шаг 2 — мягкое насыщение для сглаживания редких пиков.
    Результат всегда в диапазоне [-1, 1] без жёсткого клиппинга.
    """
    max_val = np.max(np.abs(audio))
    if max_val < 1e-9:
        return audio
    # Нормализуем до 0.9, чтобы оставить запас для tanh
    audio = audio / max_val * 0.9
    # Мягкое насыщение: tanh(x) ≈ x при малых x, плавно ограничивает пики
    audio = np.tanh(audio * 1.5) / np.tanh(1.5)
    return audio

def generate_track(notes_dict, track_id):
    total_samples = int(TRACK_DURATION * SR)
    mixed_audio = np.zeros(total_samples, dtype=np.float32)
    
    # Новый формат меток: массив формы (K, 2), где K - количество фреймов
    # Метки: [lower_fret, upper_fret]
    total_frames = int(np.ceil(TRACK_DURATION * FRAME_RATE))
    label_matrix = np.full((total_frames, 2), SILENCE_CLASS, dtype=np.int32) 
    
    # Матрица для Onset (начало удара по струне) - теперь для КАЖДОЙ струны отдельно (T, 2)
    onset_matrix = np.zeros((total_frames, 2), dtype=np.float32)

    avail_lower = list(notes_dict['lower'].keys())
    avail_upper = list(notes_dict['upper'].keys())

    current_time = 0.0

    # Выбираем ритм-шаблон для трека (только медленные кюи)
    rhythm_profile = random.choice(['kui_standard', 'slow_arpeggio', 'very_slow'])

    while current_time < TRACK_DURATION - 0.5:
        fret_lower, fret_upper = get_realistic_fret_pair(avail_lower, avail_upper)
        
        if fret_lower == SILENCE_CLASS and fret_upper == SILENCE_CLASS:
            # Пауза - просто пропускаем время
            current_time += random.uniform(0.1, 0.4)
            continue
            
        lower_variations = notes_dict['lower'].get(fret_lower, [])
        upper_variations = notes_dict['upper'].get(fret_upper, [])
        
        audio_lower = random.choice(lower_variations) if lower_variations else np.zeros(100, dtype=np.float32)
        audio_upper = random.choice(upper_variations) if upper_variations else np.zeros(100, dtype=np.float32)

        # Задержка между ударами по струнам (имитация медиатора/пальца)
        delay_sec = random.uniform(0.005, 0.02)
        if random.choice([True, False]):
            start_l, start_u = current_time + delay_sec, current_time
        else:
            start_l, start_u = current_time, current_time + delay_sec

        audios = [
            (audio_lower, start_l, 'lower', fret_lower),
            (audio_upper, start_u, 'upper', fret_upper),
        ]

        for audio, start_time, string_name, fret in audios:
            start_sample = int(start_time * SR)
            if start_sample >= total_samples:
                continue

            end_sample = start_sample + len(audio)

            if end_sample > total_samples:
                cut_length = total_samples - start_sample
                mixed_audio[start_sample:total_samples] += audio[:cut_length]
                actual_duration = cut_length / SR
            else:
                mixed_audio[start_sample:end_sample] += audio
                actual_duration = len(audio) / SR
                
            # Заполняем матрицу меток:
            # Увеличиваем label_duration, чтобы метка продолжалась пока звучит струна (до 0.8 секунд)
            label_duration = min(0.8, actual_duration)
            start_frame = int(start_time * FRAME_RATE)
            end_frame = int((start_time + label_duration) * FRAME_RATE)
            
            str_idx = 0 if string_name == 'lower' else 1
            # Записываем лад (0-20)
            if start_frame < total_frames and fret != SILENCE_CLASS:
                max_f = min(end_frame, total_frames)
                label_matrix[start_frame:max_f, str_idx] = fret
                
                # Добавляем Onset (начало ноты) в onset_matrix
                # Теперь мы записываем в конкретный столбец струны (str_idx)
                # И строго 1.0 без размытия по соседним кадрам
                onset_matrix[start_frame, str_idx] = 1.0

        # Вычисляем время до следующего извлечения звука (Увеличили интервалы для неспешности)
        if rhythm_profile == 'kui_standard':
            step = random.uniform(0.6, 0.9)
        elif rhythm_profile == 'slow_arpeggio':
            step = random.uniform(0.9, 1.4)
        else:
            # very_slow - долгие протяжные ноты с паузами
            step = random.uniform(1.4, 2.5)

        current_time += step

    mixed_audio = soft_clip(mixed_audio)

    # Сохраняем аудио 
    sf.write(os.path.join(OUTPUT_FOLDER, f"track_{track_id}.wav"), mixed_audio, SR)
    
    # Сохраняем лейблы в .npy
    np.save(os.path.join(OUTPUT_FOLDER, f"track_{track_id}_labels.npy"), label_matrix)
    np.save(os.path.join(OUTPUT_FOLDER, f"track_{track_id}_onsets.npy"), onset_matrix)

if __name__ == "__main__":
    notes_dict = load_source_notes(SOURCE_FOLDER)
    if notes_dict['lower'] and notes_dict['upper']:
        print(f"Генерация {NUM_TRACKS} треков...")
        for i in tqdm(range(NUM_TRACKS), desc="Создание датасета"):
            generate_track(notes_dict, i)
        print("\nГотово! Датасет собран и нормализован.")
    else:
        print("Папка с исходниками пуста или файлы названы неверно (нужны lower_0.wav и т.д.)")