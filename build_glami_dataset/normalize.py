"""Map the raw, per-locale pattern/length values (from extract.py) onto a shared
set of English canonical labels, purely by hand-curated lookup against the actual
distinct values observed in the manifest (see extract.py's printed dumps) - nothing
here is guessed for a value that wasn't actually seen in the data.

Values that are genuinely ambiguous as a length/pattern *category* (e.g. Spanish
"Normal" or "Corte largo" - a vague cut note, not a mini/midi/maxi-style category)
are intentionally left unmapped (-> None) rather than assigned a best guess, so
build_dataset.py drops them instead of fabricating a label.
"""

PATTERN_MAP = {
    "ee": {
        "ühevärviline": "solid",
        "lilleline": "floral",
        "loomamuster": "animal",
        "triibuline": "striped",
        "täpiline": "polka_dot",
        "kanga segu": "mixed",
        "ruuduline": "checkered",
        "trükitud logo": "logo",
        "kahevärviline": "two_tone",
        "paisley-muster": "paisley",
        "etno muster": "ethnic",
        "trükitud motiiv": "graphic",
        "motoprint": "moto_print",
        "värvigradient": "gradient",
        "orientaalse mustriga": "oriental",
        "kamuflaaž": "camo",
        "peene triibuga": "striped",
    },
    "lt": {
        "vienspalvis": "solid",
        "su žiedų   gėlių raštais": "floral",
        "languotas": "checkered",
        "dryžuotas": "striped",
        "su gyvūnų raštais": "animal",
        "taškuotas": "polka_dot",
        "mišinys": "mixed",
        "su spausdintais logotipais": "logo",
        "„paisley“ stiliaus modelis": "paisley",
        "su spausdintais šūkiais": "graphic_text",
        "spalvų blokai": "two_tone",
        "su spausdintas motyvais": "graphic",
        "spalvų gradientas": "gradient",
        "modelis su tautiniais motyvais": "ethnic",
    },
    "lv": {
        "vienkrāsas": "solid",
        "ar ziediem augiem": "floral",
        "dzīvnieku apdruka": "animal",
        "svītrains": "striped",
        "jaukta tipa": "mixed",
        "rūtots": "checkered",
        "punktots": "polka_dot",
        "logotipu apdruka": "logo",
        "lāsīšu raksts": "polka_dot",
        "krāsu gradients": "gradient",
        "motīvu apdruka": "graphic",
        "krāsu bloķēšana": "two_tone",
        "moto apdruka": "moto_print",
        "austrumu raksts": "oriental",
    },
    "si": {
        "univerzalne barve": "solid",
        "rožast": "floral",
        "cvetličen": "floral",
        "črtast": "striped",
        "živalski tisk": "animal",
        "pikčast": "polka_dot",
        "melanž": "mixed",
        "logo potisk": "logo",
        "karirast": "checkered",
        "kockast": "checkered",
        "gladek": "solid",
        "orientalski vzorec": "oriental",
        "etno vzorec": "ethnic",
        "paisley vzorec": "paisley",
    },
    "es": {
        "color liso": "solid",
        "floral": "floral",
        "mélange": "mixed",
        "a rayas": "striped",
        "estampado animal": "animal",
        "bicolor": "two_tone",
        "punteado": "polka_dot",
        "a cuadros": "checkered",
        "estampado con logo": "logo",
        "con mensaje": "graphic_text",
        "motivo paisley": "paisley",
    },
}

LENGTH_MAP = {
    "ee": {
        "lühike mini": "mini",
        "põlvepikkune": "knee_midi",
        "3 4 pikkus": "three_quarter",
        "7 8 pikkus": "seven_eighth",
        "pikk": "maxi",
    },
    "lt": {
        "trumpas mini": "mini",
        "iki kelių": "knee_midi",
        "3 4 ilgio": "three_quarter",
        "7 8 ilgio": "seven_eighth",
        "ilgas maksi": "maxi",
        "ilgas modelis": "maxi",
        "trumpas modelis": "mini",
    },
    "lv": {
        "īsi mini": "mini",
        "līdz ceļgalam": "knee_midi",
        "3 4 garums": "three_quarter",
        "7 8 garums": "seven_eighth",
        "garš maksi": "maxi",
    },
    "si": {
        "kratka mini": "mini",
        "kratka": "mini",
        "do kolen": "knee_midi",
        "3 4 dolg": "three_quarter",
        "pol dolga": "half_length",
        "7 8 dolg": "seven_eighth",
        "dolga maksi": "maxi",
        "dolga": "maxi",
        "mini": "mini",
        "midi": "knee_midi",
    },
    "es": {
        "corto   mini": "mini",
        "hasta la rodilla": "knee_midi",
        "largo de 3 4": "three_quarter",
        "largo de 7 8": "seven_eighth",
        "largo   maxi": "maxi",
    },
}


def _lookup(raw_value, table):
    if raw_value is None:
        return None
    key = raw_value.strip().lower()
    if key in table:
        return table[key]
    # tolerate minor trailing bleed from a rare/unlisted boundary label,
    # e.g. "Triibuline Zertifikate" -> still recognize the "Triibuline" prefix
    for known_key, canonical in sorted(table.items(), key=lambda kv: -len(kv[0])):
        if key.startswith(known_key):
            return canonical
    return None


def normalize_pattern(geo, raw_value):
    return _lookup(raw_value, PATTERN_MAP.get(geo, {}))


def normalize_length(geo, raw_value):
    return _lookup(raw_value, LENGTH_MAP.get(geo, {}))
