from . import cmudict

_pad = '<pad>'
_punc = list('!\'(),-.:~? ')
_SILENCES = ['sp', 'spn', 'sil']

_eng_characters = list('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz')

# arpabet WITH stress
# Prepend "@" to ARPAbet symbols to ensure uniqueness (some are the same as uppercase letters):
_arpabet = ['@' + s for s in cmudict.valid_symbols]
_a_silences = ["@" + s for s in _silences]

# arpabet WITHOUT stress
_cmu_characters = [
    'AA', 'AE', 'AH',
    'AO', 'AW', 'AY',
    'B', 'CH', 'D', 'DH', 'EH', 'ER', 'EY',
    'F', 'G', 'HH', 'IH', 'IY',
    'JH', 'K', 'L', 'M', 'N', 'NG', 'OW', 'OY',
    'P', 'R', 'S', 'SH', 'T', 'TH', 'UH', 'UW',
    'V', 'W', 'Y', 'Z', 'ZH'
]
_cmu_characters = ['@' + s for s in _cmu_characters]

# Korean jamo
_jamo_leads = "".join([chr(_) for _ in range(0x1100, 0x1113)])
_jamo_vowels = "".join([chr(_) for _ in range(0x1161, 0x1176)])
_jamo_tails = "".join([chr(_) for _ in range(0x11A8, 0x11C3)])
_jamo_trashes = "".join([chr(_) for _ in range(0x11C4, 0x1215)])
_kor_characters = list(_jamo_leads + _jamo_vowels + _jamo_tails)

# Characters to represent Chinese
_cht_characters = list('abcdefghijklmnopqrstuvwxyz12345')

# Japanese characters
_jap_romaji_characters = [
     'N', 'a', 'b', 'by', 'ch', 'cl', 'd', 'dy', 'e', 'f', 'g',
     'gy', 'h', 'hy', 'i', 'j', 'k', 'ky', 'm', 'my', 'n', 'ny',
     'o', 'p', 'pau', 'py', 'r', 'ry', 's', 'sh', 't', 'ts', 'u',
     'v', 'w', 'y', 'z'
]

_jap_kana_characters = [
     '、',
     'ぁ', 'あ', 'ぃ', 'い', 'ぅ', 'う', 'ぇ', 'え', 'ぉ', 'お',
     'か', 'が', 'き', 'ぎ', 'く', 'ぐ', 'け', 'げ', 'こ', 'ご',
     'さ', 'ざ', 'し', 'じ', 'す', 'ず', 'せ', 'ぜ', 'そ', 'ぞ',
     'た', 'だ', 'ち', 'っ', 'つ', 'づ', 'て', 'で', 'と', 'ど',
     'な', 'に', 'ぬ', 'ね', 'の', 'は', 'ば', 'ぱ', 'ひ', 'び',
     'ぴ', 'ふ', 'ぶ', 'ぷ', 'へ', 'べ', 'ぺ', 'ほ', 'ぼ', 'ぽ',
     'ま', 'み', 'む', 'め', 'も', 'ゃ', 'や', 'ゅ', 'ゆ', 'ょ',
     'よ', 'ら', 'り', 'る', 'れ', 'ろ', 'わ', 'を', 'ん', 'ゔ',
     'ー'
]

# Export all symbols:
# refer to https://pms.maum.ai/confluence/x/hJgjAg for list of symbol sets.

## English
# eng, eng2 (use arpabet WITH stress)
eng_symbols = _pad + _eng_characters + _punc + _arpabet + _a_silences
# cmu (use arpabet WITHOUT stress)
cmu_symbols = _pad + _eng_characters + _punc + _cmu_characters + _a_silences

## Korean
# kor
kor_symbols = _pad + _kor_characters + _punc + _SILENCES + list(_jamo_trashes)

# Chinese
# cht
cht_symbols = _pad + _cht_characters + _punc

# Japanese
# jap, jap_romaji
jap_romaji_symbols = _pad + _jap_romaji_characters + _punc
# jap_kana
jap_kana_symbols = _pad + _jap_kana_characters + _punc
