"""Gold-independent answer extraction; content, format and stopping stay separate."""
import re
import unicodedata


def normalize(text):
    text = unicodedata.normalize('NFKC', text).casefold()
    return ' '.join(''.join(c if c.isalnum() or c.isspace() else ' ' for c in text).split())


def extract_answer(text, choices):
    if len(choices) != 4:
        raise ValueError('Exactly four choices required')
    s = text.strip()
    if not s:
        return {'prediction': None, 'route': 'empty'}
    s = re.sub(r'\*\*([^*]+)\*\*', r'\1', s)
    s = re.sub(r'^(?:the\s+)?(?:correct\s+)?(?:answer|option|choice)\s*(?:is\s+|:\s*)', '', s, flags=re.I)
    m = re.match(r'^\(?([ABCD])\)?(?=$|\s|[.,:;\]])', s)
    options = [normalize(c) for c in choices]
    if m:
        pred = 'ABCD'.index(m.group(1))
        tail = s[m.end():].lstrip(' .,:;)-]')
        if re.match(r'^(?:or|and|/|&)\s*\(?[ABCD]\b', tail, flags=re.I):
            return {'prediction': None, 'route': 'ambiguous_letters'}
        body = normalize(tail)
        matches = [i for i, option in enumerate(options) if option and (body == option or body.startswith(option + ' '))]
        if matches:
            longest = max(len(options[i]) for i in matches)
            strongest = [i for i in matches if len(options[i]) == longest]
            if pred not in strongest:
                return {'prediction': None, 'route': 'letter_option_conflict'}
        return {'prediction': pred, 'route': 'letter'}
    exact = [i for i, option in enumerate(options) if normalize(s) == option]
    if len(exact) == 1:
        return {'prediction': exact[0], 'route': 'exact_option_text'}
    return {'prediction': None, 'route': 'ambiguous_option_text' if len(exact) > 1 else 'unparseable'}


def decode_output(ids, tokenizer, stop_ids, choices, gold, max_new_tokens):
    stop = next((i for i, token in enumerate(ids) if token in stop_ids), None)
    content = ids if stop is None else ids[:stop]
    text = tokenizer.decode(content, skip_special_tokens=False)
    result = extract_answer(text, choices)
    strict_valid = text.strip() in 'ABCD' and len(text.strip()) == 1
    result.update(text=text, content_ids=content, generated_ids=ids if stop is None else ids[:stop + 1],
                  correct=result['prediction'] == gold, valid_answer=result['prediction'] is not None,
                  exact_letter_format=strict_valid, strict_correct=strict_valid and text.strip() == 'ABCD'[gold],
                  terminated=stop is not None, hit_length_cap=stop is None and len(ids) >= max_new_tokens,
                  completed_correct=stop is not None and result['prediction'] == gold,
                  generated_tokens=len(ids) if stop is None else stop + 1)
    return result
