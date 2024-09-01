# pip install musiclang_predict
from musiclang_predict import MusicLangPredictor

nb_tokens = 1024
temperature = 0.9  # Don't go over 1.0, at your own risks !
top_p = 1.0  # <=1.0, Usually 1 best to get not too much repetitive music
seed = 16  # change here to change result, or set to 0 to unset seed

ml = MusicLangPredictor('musiclang/musiclang-v2')  # Only available model for now

score = ml.predict(
    nb_tokens=nb_tokens,  # 1024 tokens ~ 25s of music (depending of the number of instruments generated)
    temperature=temperature,
    topp=top_p,
    rng_seed=seed  # change here to change result, or set to 0 to unset seed
)
score.to_midi('outputs/test.mid')  # Open that file in your favourite DAW, score editor or even in VLC
