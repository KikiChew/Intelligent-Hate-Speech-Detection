"""Robust internet-slang expansion and disguise normalization (Part B v2).

Handles:
1. Standard internet acronyms and contractions (u -> you, stfu -> shut the fuck up, fr -> for real).
2. Character elongation / repeated letters (fuckassss -> fuckass, fuuuuck -> fuck, shiiiiit -> shit).
3. Obfuscated / masked profanities (f*ck, f.u.c.k, b!tch, sh!t, a$$, @ss).
4. Compound profanities out-of-vocabulary in downstream TF-IDF (fuckass -> fuck ass, dumbfuck -> dumb fuck).
"""
import re
import unicodedata

SLANG = {
    # Pronouns, casual chat, contractions
    'u': 'you',
    'ya': 'you',
    'yall': 'you all',
    'yalll': 'you all',
    'im': 'i am',
    'ive': 'i have',
    'ill': 'i will',
    'dont': 'do not',
    'doesnt': 'does not',
    'didnt': 'did not',
    'cant': 'cannot',
    'wont': 'will not',
    'isnt': 'is not',
    'arent': 'are not',
    'wasnt': 'was not',
    'werent': 'were not',
    'couldnt': 'could not',
    'wouldnt': 'would not',
    'shouldnt': 'should not',
    'gonna': 'going to',
    'wanna': 'want to',
    'gotta': 'got to',
    'lemme': 'let me',
    'gimme': 'give me',
    'kinda': 'kind of',
    'sorta': 'sort of',
    'cuz': 'because',
    'coz': 'because',
    'cos': 'because',
    'bc': 'because',
    'bcz': 'because',
    'bcos': 'because',
    'tho': 'though',
    'thru': 'through',
    'pls': 'please',
    'plz': 'please',
    'plss': 'please',
    'thx': 'thanks',
    'thnx': 'thanks',
    'ty': 'thank you',
    'tysm': 'thank you so much',
    'yw': 'you are welcome',
    'np': 'no problem',
    'nvm': 'never mind',
    'nm': 'not much',
    'sry': 'sorry',
    'soz': 'sorry',
    'ppl': 'people',
    'msg': 'message',
    'msgs': 'messages',
    'dm': 'direct message',
    'dms': 'direct messages',
    'pm': 'private message',
    'irl': 'in real life',
    'tbh': 'to be honest',
    'ngl': 'not going to lie',
    'imo': 'in my opinion',
    'imho': 'in my humble opinion',
    'idk': 'i do not know',
    'idc': 'i do not care',
    'ik': 'i know',
    'ikr': 'i know right',
    'afaik': 'as far as i know',
    'fyi': 'for your information',
    'btw': 'by the way',
    'btdubs': 'by the way',
    'asap': 'as soon as possible',
    'brb': 'be right back',
    'bbl': 'be back later',
    'gtg': 'got to go',
    'g2g': 'got to go',
    'ttyl': 'talk to you later',
    'ttys': 'talk to you soon',
    'cya': 'see you',
    'cu': 'see you',
    'gn': 'good night',
    'gm': 'good morning',
    'hru': 'how are you',
    'hbu': 'how about you',
    'wbu': 'what about you',
    'wyd': 'what are you doing',
    'wya': 'where are you at',
    'wym': 'what do you mean',
    'wdym': 'what do you mean',
    'wth': 'what the hell',
    'wtf': 'what the fuck',
    'omg': 'oh my god',
    'omfg': 'oh my fucking god',
    'omw': 'on my way',
    'rn': 'right now',
    'atm': 'at the moment',
    'tmi': 'too much information',
    'fomo': 'fear of missing out',
    'iykyk': 'if you know you know',
    'smh': 'shaking my head',
    'smfh': 'shaking my fucking head',
    'lol': 'laughing out loud',
    'lmao': 'laughing my ass off',
    'lmfao': 'laughing my fucking ass off',
    'rofl': 'rolling on the floor laughing',
    'roflmao': 'rolling on the floor laughing my ass off',
    'jk': 'just kidding',
    'jkjk': 'just kidding',
    'fr': 'for real',
    'frfr': 'for real for real',
    'istg': 'i swear to god',
    'ong': 'on god',
    'sus': 'suspicious',
    'afk': 'away from keyboard',
    'gg': 'good game',
    'ggwp': 'good game well played',
    'glhf': 'good luck have fun',
    'lmk': 'let me know',
    'hmu': 'hit me up',
    'ily': 'i love you',
    'ilysm': 'i love you so much',
    'ily2': 'i love you too',
    'ilyt': 'i love you too',
    'bff': 'best friends forever',
    'bf': 'boyfriend',
    'gf': 'girlfriend',
    'ootd': 'outfit of the day',
    'tbt': 'throwback thursday',
    'pov': 'point of view',
    'grwm': 'get ready with me',
    'fyp': 'for you page',
    'icymi': 'in case you missed it',
    'nsfw': 'not safe for work',
    'sfw': 'safe for work',
    'ama': 'ask me anything',
    'eli5': 'explain like i am five',
    'tl;dr': 'too long did not read',
    'tldr': 'too long did not read',
    'faq': 'frequently asked questions',
    'op': 'original poster',
    'oc': 'original content',
    'sm': 'social media',
    'pfp': 'profile picture',
    'dp': 'display picture',

    # Offensive / harmful expressions
    'stfu': 'shut the fuck up',
    'gtfo': 'get the fuck out',
    'gfy': 'go fuck yourself',
    'fyou': 'fuck you',
    'fuk': 'fuck',
    'fck': 'fuck',
    'fk': 'fuck',
    'fkn': 'fucking',
    'fkin': 'fucking',
    'fcking': 'fucking',
    'fuking': 'fucking',
    'mf': 'motherfucker',
    'mfer': 'motherfucker',
    'mofo': 'motherfucker',
    'bs': 'bullshit',
    'bullsh*t': 'bullshit',
    'pos': 'piece of shit',
    'sob': 'son of a bitch',
    'b1tch': 'bitch',
    'biatch': 'bitch',
    'bish': 'bitch',
    'a**hole': 'asshole',
    'ahole': 'asshole',
    'azzhole': 'asshole',
    'dumbass': 'dumb ass',
    'kys': 'kill yourself',
    'killyourself': 'kill yourself',
    'kms': 'kill myself',
    'af': 'as fuck',
    'tf': 'the fuck',
    'wttf': 'what the fuck',
    'fml': 'fuck my life',
    'nb': 'noob',
    'n00b': 'noob',
}

# Compound profanities that should be segmented into constituent tokens
# so downstream TF-IDF and word embeddings reliably detect them.
COMPOUND_SWEARS = {
    'fuckass': 'fuck ass',
    'bitchass': 'bitch ass',
    'dumbass': 'dumb ass',
    'fatass': 'fat ass',
    'smartass': 'smart ass',
    'jackass': 'jack ass',
    'hardass': 'hard ass',
    'lazyass': 'lazy ass',
    'brokeass': 'broke ass',
    'punkass': 'punk ass',
    'uglyass': 'ugly ass',
    'crazyass': 'crazy ass',
    'stupidass': 'stupid ass',
    'asshat': 'ass hat',
    'assclown': 'ass clown',
    'asswipe': 'ass wipe',
    'assface': 'ass face',
    'asshead': 'ass head',
    'dumbfuck': 'dumb fuck',
    'fuckhead': 'fuck head',
    'fuckface': 'fuck face',
    'fucktard': 'fuck retard',
    'fuckboy': 'fuck boy',
    'fuckboi': 'fuck boy',
    'dickhead': 'dick head',
    'dipshit': 'dip shit',
    'shithead': 'shit head',
    'shitbag': 'shit bag',
    'shitface': 'shit face',
    'horseshit': 'horse shit',
    'douchebag': 'douche bag',
    'scumbag': 'scum bag',
    'motherfucker': 'mother fucker',
    'motherfucking': 'mother fucking',
}

# Words where consecutive duplicate characters should be preserved
# (e.g., 'ass' vs 'as', 'hell' vs 'hel', 'kill' vs 'kil', 'noob' vs 'nob').
KNOWN_DOUBLES = {
    'ass', 'hell', 'kill', 'noob', 'cool', 'good', 'too', 'see', 'free',
    'beer', 'bitchass', 'dumbass', 'fatass', 'fuckass', 'jackass', 'smartass',
    'grass', 'pass', 'class', 'mass', 'less', 'mess', 'miss', 'kiss', 'piss',
    'boss', 'loss', 'full', 'bull', 'call', 'fall', 'wall', 'all', 'will',
    'tell', 'well', 'feed', 'need', 'seed', 'weed', 'deep', 'keep', 'sleep',
    'book', 'look', 'took', 'foot', 'root', 'boot', 'fool', 'pool', 'tool',
}

_ZERO_WIDTH = re.compile(r'[\u200b\u200c\u200d\u2060\ufeff]')
_MASK = re.compile(r'(?<=[a-zA-Z0-9])[*!._\-]+(?=[a-zA-Z0-9])')
_TERMS = sorted(SLANG, key=len, reverse=True)
_PATTERN = re.compile(r'(?<!\w)(?:' + '|'.join(re.escape(x) for x in _TERMS) + r')(?!\w)', re.IGNORECASE)


def _reduce_elongation(token):
    """Reduce runs of 3+ repeated characters while preserving valid English double-letters."""
    def _repl(match):
        char = match.group(1)
        c2 = char * 2
        c1 = char
        full_c2 = re.sub(re.escape(match.group(0)), c2, token.lower())
        # If preserving double letter matches a known double word/suffix (e.g. -ass, -hell, noob)
        if any(full_c2.endswith(d) or full_c2 == d for d in KNOWN_DOUBLES):
            return c2
        if char.lower() in 'eols' and any(full_c2.endswith(w) for w in ('ass', 'hell', 'noob', 'piss')):
            return c2
        return c1

    return re.sub(r'([a-zA-Z])\1{2,}', _repl, token)


def normalize_slang(text):
    """Robustly normalize text, internet slang, disguised words, and elongation.

    1. Normalizes Unicode and removes zero-width characters.
    2. Removes common obfuscation masks (*, ., -, _) inside words (f*ck -> fuck).
    3. Handles leetspeak substitutions (@ -> a, $ -> s, 1/! inside word -> i).
    4. Reduces character elongation (fuckassss -> fuckass, fuuuuck -> fuck).
    5. Segments compound profanities (fuckass -> fuck ass, dumbfuck -> dumb fuck).
    6. Expands internet abbreviations and acronyms via word-boundary matching.
    """
    if text is None:
        return ''

    # 1. Unicode & zero-width normalization
    raw = unicodedata.normalize('NFKC', str(text))
    raw = _ZERO_WIDTH.sub('', raw)

    # 2. Leet substitutions inside words
    raw = re.sub(r'@(?=[a-zA-Z])|(?<=[a-zA-Z])@', 'a', raw)
    raw = raw.replace('$$', 'ss').replace('$s', 'ss').replace('s$', 'ss')
    raw = re.sub(r'\$(?=[a-zA-Z])|(?<=[a-zA-Z])\$', 's', raw)
    raw = re.sub(r'(?<=[a-zA-Z])[!1](?=[a-zA-Z])', 'i', raw)
    raw = re.sub(r'(?<=[a-zA-Z])00(?=[a-zA-Z])', 'oo', raw)
    raw = re.sub(r'(?<=[a-zA-Z])0(?=[a-zA-Z])', 'o', raw)

    # 3. Delimiter masking between letters/digits: f*ck -> fck (or fuck)
    raw = _MASK.sub('', raw)

    # 4. Token-level elongation & compound swear splitting
    words = raw.split()
    processed_words = []
    for word in words:
        # Separate leading and trailing punctuation
        m_pref = re.match(r'^\W+', word)
        m_suff = re.search(r'\W+$', word)
        prefix = m_pref.group(0) if m_pref else ''
        suffix = m_suff.group(0) if m_suff else ''
        core = word[len(prefix):len(word) - len(suffix)] if suffix else word[len(prefix):]

        if core:
            # Elongation reduction: fuckassss -> fuckass
            core = _reduce_elongation(core)
            lower_core = core.lower()

            # Compound profanity splitting: fuckass -> fuck ass
            if lower_core in COMPOUND_SWEARS:
                core = COMPOUND_SWEARS[lower_core]

        processed_words.append(prefix + core + suffix)

    reconstructed = ' '.join(processed_words)

    # 5. Whole-word slang dictionary expansion
    return _PATTERN.sub(lambda m: SLANG[m.group(0).lower()], reconstructed)


def slang_stats():
    """Return dictionary size and compound profanity count for documentation."""
    return len(SLANG) + len(COMPOUND_SWEARS)
