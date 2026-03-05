import json
import os
import random
import numpy as np
import soundfile as sf
import librosa
from tqdm import tqdm
from collections import defaultdict
import argparse
import pandas as pd


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return super(NumpyEncoder, self).default(obj)


# ==========================================
# 1. global settings
# ==========================================
parser = argparse.ArgumentParser(description="Bird Audio Dataset Generator")
parser.add_argument('--output_root', type=str, required=True,
                    help='Path to save the generated dataset')
parser.add_argument('--overlap_ratio', type=float, default=0.2,
                    help='Target overlap ratio (e.g., 0.2 for 20%)')
parser.add_argument('--input_parquet', type=str,
                    default="./pure_bird_manifests/", help='Path to input Parquet manifest')

args = parser.parse_args()

INPUT_PARQUET = args.input_parquet
OUTPUT_ROOT = args.output_root
TARGET_OVERLAP_RATIO = args.overlap_ratio

# hyperparameters setting
TOP_N = 10
TARGET_TRAIN_HOURS = 320
TARGET_DEV_HOURS = 80
TARGET_TEST_HOURS = 100

MIN_SESSION_DURATION = 10 * 60  # duration
MIN_SPEAKERS = 2
MAX_SPEAKERS = 5
MAX_CONCURRENT_LIMIT = 4

SILENCE_MIN = 0.1
SILENCE_MAX = 0.5
SEED = 1744

MASTER_SR = 32000  # sampling rates
TARGET_SRS = [16000, 32000]

os.umask(0)

# ==========================================
# 2. 辅助函数
# ==========================================


def load_audio_multi_sr(path, target_srs, offset=0.0, duration=None):
    results = {}
    try:
        for sr in target_srs:

            y, _ = librosa.load(path, sr=sr, mono=True,
                                offset=offset, duration=duration)
            results[sr] = y
        return results
    except Exception:
        return None


def filter_top_n_species(data_list, n=10):
    print(" Counting and filtering Top N bird species...")
    species_durations = defaultdict(float)
    species_clips = defaultdict(list)

    for item in data_list:
        code = item.get('ebird_code')
        if not code:
            continue
        species_clips[code].append(item)
        dur = 0.0
        events = item.get('detected_events')
        if isinstance(events, (list, np.ndarray)) and len(events) > 0:
            for e in events:
                if len(e) >= 2 and e[1] > e[0]:
                    dur += (e[1] - e[0])
        else:
            dur = 10.0
        species_durations[code] += dur

    sorted_species = sorted(species_durations.items(),
                            key=lambda x: x[1], reverse=True)
    top_n_species = [x[0] for x in sorted_species[:n]]
    print(f"Top {n} species selected for BirdDiar-XC")

    filtered_data = []
    for sp in top_n_species:
        filtered_data.extend(species_clips[sp])
    return filtered_data, top_n_species

# ==========================================
# the class for data generator
# ==========================================


class LargeScaleGenerator:
    def __init__(self, clip_pool, species_list):
        self.clip_pool = clip_pool  # the Parquet data
        self.all_species = list(species_list)
        self.sp_to_clips = defaultdict(list)
        for c in clip_pool:
            self.sp_to_clips[c['ebird_code']].append(c)

    def get_random_clip(self, species):
        if not self.sp_to_clips[species]:
            return None
        return random.choice(self.sp_to_clips[species])

    def generate_session(self, session_idx, output_base_dir, split_name):
        # the account among different number of speakers
        speaker_range = list(range(MIN_SPEAKERS, MAX_SPEAKERS + 1))
        weight_map = {2: 0.3, 3: 0.3, 4: 0.3, 5: 0.1}

        current_weights = [weight_map.get(s, 0.25) for s in speaker_range]

        num_speakers = random.choices(
            speaker_range, weights=current_weights, k=1)[0]

        available_species = [
            s for s in self.all_species if len(self.sp_to_clips[s]) > 0]
        active_species = random.sample(available_species, min(
            len(available_species), num_speakers))

        # num_speakers = random.randint(MIN_SPEAKERS, MAX_SPEAKERS)
        # available_species = [s for s in self.all_species if len(self.sp_to_clips[s]) > 0]
        # active_species = random.sample(available_species, min(len(available_species), num_speakers))

        est_len_master = int(MIN_SESSION_DURATION * 1.5 * MASTER_SR)
        session_audios = {sr: np.zeros(
            int(est_len_master * (sr / MASTER_SR)), dtype=np.float32) for sr in TARGET_SRS}
        occupancy_mask = np.zeros(est_len_master, dtype=np.int8)

        session_events = []
        max_filled_master = 0

        while max_filled_master < (MIN_SESSION_DURATION * MASTER_SR):
            current_spk = random.choice(active_species)
            clip = self.get_random_clip(current_spk)
            if not clip:
                continue

            # get the dense event part
            d_start = clip['densest_start']
            d_end = clip['densest_end']

            clips_dict = load_audio_multi_sr(
                clip['audio_path'], TARGET_SRS, offset=d_start, duration=5.0)
            if not clips_dict:
                continue
            # ================= [新增：随机增益增强逻辑] =================
            delta_db = random.uniform(-3, 3)
            gain_factor = 10 ** (delta_db / 20)

            for sr in TARGET_SRS:
                # fade in and out
                y_aug = clips_dict[sr] * gain_factor
                clips_dict[sr] = self.apply_fades(y_aug, sr, fade_ms=50)
            # ===========================================================
            y_m = clips_dict[MASTER_SR]

            clip_len_sec = len(y_m) / MASTER_SR
            clip_len_samples = len(y_m)

            raw_events = clip.get('detected_events', [])
            current_clip_events = []
            if isinstance(raw_events, (list, np.ndarray)) and len(raw_events) > 0:
                for e in raw_events:

                    if e[1] > d_start and e[0] < d_end:

                        rel_s = max(0.0, e[0] - d_start)
                        rel_e = min(clip_len_sec, e[1] - d_start)
                        if rel_e > rel_s:
                            current_clip_events.append((rel_s, rel_e))

            # back tracking
            current_overlap_samples = np.count_nonzero(
                occupancy_mask[:max_filled_master] > 1)
            current_ratio = current_overlap_samples / \
                max_filled_master if max_filled_master > 0 else 0

            start_pos_m = max_filled_master
            placed = False

            if (current_ratio < TARGET_OVERLAP_RATIO and max_filled_master > MASTER_SR * 5):
                for _ in range(20):
                    back_limit = int(
                        min(max_filled_master * 0.9, 60 * MASTER_SR))
                    proposed_start = random.randint(
                        max(0, max_filled_master - back_limit), max_filled_master)

                    check_region = occupancy_mask[proposed_start:
                                                  proposed_start + clip_len_samples]
                    if len(check_region) > 0 and np.max(check_region) < MAX_CONCURRENT_LIMIT:
                        start_pos_m = proposed_start
                        placed = True
                        break

            if not placed:
                sil_sec = random.uniform(SILENCE_MIN, SILENCE_MAX)
                start_pos_m = max_filled_master + int(sil_sec * MASTER_SR)

            end_pos_m = start_pos_m + clip_len_samples

            if end_pos_m > len(occupancy_mask):
                pad_size = end_pos_m - len(occupancy_mask) + MASTER_SR * 60
                occupancy_mask = np.pad(occupancy_mask, (0, int(pad_size)))
                for sr in TARGET_SRS:
                    session_audios[sr] = np.pad(
                        session_audios[sr], (0, int(pad_size * (sr / MASTER_SR))))

            start_time_sec = start_pos_m / MASTER_SR
            if current_clip_events:
                for ev_s, ev_e in current_clip_events:
                    ev_start = start_pos_m + int(ev_s * MASTER_SR)
                    ev_end = start_pos_m + int(ev_e * MASTER_SR)
                    occupancy_mask[ev_start:ev_end] += 1

                    session_events.append({
                        "label": current_spk,
                        "start": round(start_time_sec + ev_s, 4),
                        "end": round(start_time_sec + ev_e, 4)
                    })
            else:

                occupancy_mask[start_pos_m:end_pos_m] += 1
                session_events.append({
                    "label": current_spk,
                    "start": round(start_time_sec, 4),
                    "end": round(start_time_sec + clip_len_sec, 4)
                })

            # audio signal addition
            for sr in TARGET_SRS:
                y = clips_dict[sr]
                sp_sr = int(start_time_sec * sr)
                ep_sr = sp_sr + len(y)
                # session_audios[sr][sp_sr:ep_sr] += y

                if ep_sr <= len(session_audios[sr]):
                    session_audios[sr][sp_sr:ep_sr] += y
            max_filled_master = max(max_filled_master, end_pos_m)

        # save audio
        ov_tag = int(TARGET_OVERLAP_RATIO * 100)
        final_dur = max_filled_master / MASTER_SR
        base_name = f"{split_name}_ov{ov_tag}_{session_idx:05d}"

        actual_mask = occupancy_mask[:max_filled_master]
        actual_ov = np.count_nonzero(
            actual_mask > 1) / max_filled_master if max_filled_master > 0 else 0
        max_conc = int(np.max(actual_mask)) if len(actual_mask) > 0 else 0
        # ===============
        for sr in TARGET_SRS:
            max_val = np.max(np.abs(session_audios[sr]))
            if max_val > 1.0:
                session_audios[sr] = session_audios[sr] / (max_val + 1e-6)

        self._save_outputs(session_audios, actual_ov, max_conc, num_speakers,
                           session_events, final_dur, base_name, output_base_dir)
        return final_dur

    def _save_outputs(self, audios, actual_ov, max_concurrent, num_spk, events, duration, base_name, output_dir):
        max_val = np.max([np.max(np.abs(a)) for a in audios.values()])
        scale = 0.9 / max_val if max_val > 1.0 else 1.0

        for sr, y in audios.items():
            wav_dir = os.path.join(output_dir, f"wav{sr//1000}k")
            os.makedirs(wav_dir, exist_ok=True)
            cut_y = y[:int(duration * sr)] * scale
            sf.write(os.path.join(wav_dir, f"{base_name}.wav"), cut_y, sr)

        json_dir = os.path.join(output_dir, "json")
        os.makedirs(json_dir, exist_ok=True)

        # metadata
        meta = {
            "filename": base_name,
            "duration_sec": duration,
            "num_speakers": num_spk,
            "max_concurrent": max_concurrent,
            "actual_overlap": round(actual_ov, 4),
            "events": sorted(events, key=lambda x: x['start'])
        }
        with open(os.path.join(json_dir, f"{base_name}.json"), 'w') as f:
            json.dump(meta, f, indent=2, cls=NumpyEncoder)

    def apply_fades(self, y, sr, fade_ms=50):

        fade_samples = int(fade_ms * sr / 1000)
        if len(y) < 2 * fade_samples:
            return y

        fade_in = np.linspace(0, 1, fade_samples)

        fade_out = np.linspace(1, 0, fade_samples)

        y_faded = y.copy()
        y_faded[:fade_samples] *= fade_in
        y_faded[-fade_samples:] *= fade_out

        return y_faded
# ==========================================
# main function
# ==========================================


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    df = pd.read_parquet(INPUT_PARQUET)
    raw_data = df.to_dict('records')
    filtered_data, top_species = filter_top_n_species(raw_data, n=TOP_N)

    for split, hours in [("train", TARGET_TRAIN_HOURS), ("dev", TARGET_DEV_HOURS), ("test", TARGET_TEST_HOURS)]:
        split_dir = os.path.join(OUTPUT_ROOT, split)
        gen = LargeScaleGenerator(filtered_data, top_species)
        current_sec, idx = 0, 0
        pbar = tqdm(total=int(hours * 3600), desc=f"Generating {split}")
        while current_sec < hours * 3600:
            idx += 1
            dur = gen.generate_session(idx, split_dir, split)
            current_sec += dur
            pbar.update(int(dur))
        pbar.close()


if __name__ == "__main__":
    main()
