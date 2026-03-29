def diarize_audio(wav_path: str, n_speakers: int = None):
    """
    Returns list of (start, end, speaker_id) segments.
    If resemblyzer/sklearn not installed or fails, return None.
    """
    try:
        from resemblyzer import VoiceEncoder, preprocess_wav
        import numpy as np
        from sklearn.cluster import SpectralClustering
    except ImportError:
        return None

    try:
        wav = preprocess_wav(wav_path)  # returns np.array, 16k
        encoder = VoiceEncoder()
        sr = 16000
        win = int(1.5 * sr)
        hop = int(0.75 * sr)
        embeds = []
        timestamps = []
        for start in range(0, len(wav) - win + 1, hop):
            chunk = wav[start:start+win]
            emb = encoder.embed_utterance(chunk)
            embeds.append(emb)
            timestamps.append((start/sr, (start+win)/sr))
        if len(embeds) == 0:
            return None
        X = np.vstack(embeds)
        if n_speakers is None:
            n_speakers = min(5, max(1, int(len(X)**0.5)))
        clustering = SpectralClustering(n_clusters=n_speakers, affinity="nearest_neighbors").fit(X)
        labels = clustering.labels_
        segments = []
        for (st, en), lab in zip(timestamps, labels):
            segments.append({"start": st, "end": en, "speaker": int(lab)})
        return segments
    except Exception:
        return None
