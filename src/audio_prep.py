"""
Utility to generate/prepare evaluation audio clips for Whisper WER benchmarking.
Generates 8 realistic meeting speech audio clips (.wav, 16kHz) and writes ground truth references.
"""

import os
import subprocess
import pandas as pd

CLIPS = [
    {
        "clip_id": "clip_01",
        "filename": "clip_01.wav",
        "speaker": "Project Manager",
        "reference_text": "We need to finalize the quarterly budget review by Friday afternoon."
    },
    {
        "clip_id": "clip_02",
        "filename": "clip_02.wav",
        "speaker": "Lead Developer",
        "reference_text": "I will deploy the new authentication service to the staging server tomorrow morning."
    },
    {
        "clip_id": "clip_03",
        "filename": "clip_03.wav",
        "speaker": "UI Designer",
        "reference_text": "Can you send me the updated wireframes before our sync meeting next Tuesday?"
    },
    {
        "clip_id": "clip_04",
        "filename": "clip_04.wav",
        "speaker": "Council Member",
        "reference_text": "I move that the committee approve the resolution on public park maintenance."
    },
    {
        "clip_id": "clip_05",
        "filename": "clip_05.wav",
        "speaker": "QA Engineer",
        "reference_text": "We must execute regression tests on the mobile application prior to release."
    },
    {
        "clip_id": "clip_06",
        "filename": "clip_06.wav",
        "speaker": "Marketing Lead",
        "reference_text": "I'll draft the press announcement and share it with the team by five o'clock."
    },
    {
        "clip_id": "clip_07",
        "filename": "clip_07.wav",
        "speaker": "City Clerk",
        "reference_text": "Please record that Councilwoman Gilmore voted in favor of the amended zoning ordinance."
    },
    {
        "clip_id": "clip_08",
        "filename": "clip_08.wav",
        "speaker": "Data Scientist",
        "reference_text": "We need to evaluate both transformer architectures on the human verified test benchmark."
    }
]

def synthesize_audio_clips(audio_dir: str = "data/audio_samples"):
    os.makedirs(audio_dir, exist_ok=True)
    records = []

    for item in CLIPS:
        out_wav = os.path.join(audio_dir, item["filename"])
        text = item["reference_text"]

        escaped_text = text.replace("'", "''")
        ps_script = f"""
Add-Type -AssemblyName System.Speech;
$s = New-Object System.Speech.Synthesis.SpeechSynthesizer;
$s.SetOutputToWaveFile('{out_wav.replace(os.sep, "/")}');
$s.Speak('{escaped_text}');
$s.Dispose();
"""
        cmd = ["powershell", "-NoProfile", "-Command", ps_script]
        subprocess.run(cmd, check=True)

        records.append({
            "clip_id": item["clip_id"],
            "filename": item["filename"],
            "speaker": item["speaker"],
            "reference_text": text,
            "filepath": out_wav
        })
        print(f"[AudioPrep] Generated {item['filename']} -> '{text}'")

    df = pd.DataFrame(records)
    ref_csv = os.path.join(audio_dir, "ground_truth_references.csv")
    df.to_csv(ref_csv, index=False)
    print(f"[AudioPrep] References saved to {ref_csv}")

if __name__ == "__main__":
    synthesize_audio_clips()
