"""Add an English `description` field to GLAMI-1M-dresses-pattern-length/manifest.json,
for use as the fine-tuning caption text.

The existing `name` field is a raw product title in one of 5 locales (ee/lt/lv/si/es):
  BRAND [GARMENT-TYPE WORDS] ['MODEL NAME'] COLOR WORD(S)

Brand and model name are proper nouns (already Latin-script / international) and are
kept as-is. Garment-type and color words are locale text, translated via curated
per-locale dictionaries built from the full observed vocabulary (see word-frequency
dump in this file's history / commit message). pattern_label / length_label are
already clean English categorical labels; the generated description states them
explicitly so both attributes are unambiguous for contrastive fine-tuning.

Run from repo root: python3 build_glami_dataset/add_english_descriptions.py
"""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "datasets" / "GLAMI-1M-dresses-pattern-length" / "manifest.json"

# ---------------------------------------------------------------- garment type
# Ordered (longest/most-specific substring first) per locale. Matched against the
# lowercased name with quoted model names removed. First hit wins.
GARMENT_RULES = {
    "es": [
        ("vestido de punto", "knit dress"),
        ("vestido camisero", "shirt dress"),
        ("vestido", "dress"),
        ("túnica", "tunic"),
        ("tunica", "tunic"),
        ("blusa", "blouse"),
        ("camiseta", "t-shirt dress"),
        ("sudadera", "sweatshirt dress"),
    ],
    "lt": [
        ("palaidinės tipo suknelė", "blouse-style dress"),
        ("vasarinė suknelė", "summer dress"),
        ("megzta suknelė", "knit dress"),
        ("korsetinė suknelė", "corset dress"),
        ("kokteilinė suknelė", "cocktail dress"),
        ("sportinė suknelė", "sports dress"),
        ("suknelė", "dress"),
        ("megztinis", "knitwear dress"),
        ("tunika", "tunic"),
        ("marškinėliai", "t-shirt dress"),
        ("dirndlis", "dirndl-style dress"),
        ("kleid", "dress"),
        ("palaidinė", "blouse"),
    ],
    "si": [
        ("dolga srajca", "long shirt dress"),
        ("poletna obleka", "summer dress"),
        ("pletena obleka", "knit dress"),
        ("koktejl obleka", "cocktail dress"),
        ("večerna obleka", "evening dress"),
        ("obleka", "dress"),
        ("srajca", "shirt dress"),
        ("kombinezon", "jumpsuit"),
    ],
    "lv": [
        ('"dirndl" stila kleita', "dirndl-style dress"),
        ("adīta kleita", "knitted dress"),
        ("kokteiļkleita", "cocktail dress"),
        ("sporta kleita", "sports dress"),
        ("blūžkleita", "blouse dress"),
        ("kleita", "dress"),
        ("kombinezons", "jumpsuit"),
        ("džemperis", "sweater dress"),
        ("kleid", "dress"),
    ],
    "ee": [
        ("suvekleit", "summer dress"),
        ("kokteilikleit", "cocktail dress"),
        ("spordikleit", "sports dress"),
        ("rannakleit", "beach dress"),
        ("õhtukleit", "evening dress"),
        ("tuppkleit", "shift dress"),
        ("särkkleit", "shirt dress"),
        ("dressipluus", "sweatshirt dress"),
        ("traksiseelik", "pinafore dress"),
        ("kleit", "dress"),
        ("kleid", "dress"),
    ],
}

GARMENT_MODIFIER = {
    "es": [("de punto", "knit"), ("de verano", "summer"), ("verano", "summer"), ("de gala", "gala")],
    "lt": [("megzta", "knit"), ("vasarinė", "summer")],
    "si": [("pletena", "knit"), ("poletna", "summer")],
    "lv": [("adīta", "knitted"), ("vasaras", "summer")],
    "ee": [("kootud", "knit")],
}

# --------------------------------------------------------------------- colors
# base color word -> english
COLOR_BASE = {
    "es": {
        "negro": "black", "blanco": "white", "azul": "blue", "rojo": "red",
        "gris": "grey", "verde": "green", "beige": "beige", "rosa": "pink",
        "marrón": "brown", "amarillo": "yellow", "naranja": "orange", "orange": "orange",
        "lila": "lilac", "malva": "mauve", "crema": "cream", "caqui": "khaki",
        "navy": "navy", "denim": "denim blue", "turquesa": "turquoise",
        "dorado": "gold", "vino": "wine-red", "menta": "mint", "arena": "sand",
        "oliva": "olive", "champán": "champagne", "cereza": "cherry-red",
        "taupe": "taupe", "cognac": "cognac", "kiwi": "kiwi-green",
        "jade": "jade", "melón": "melon", "madder": "madder-red",
        "damson": "damson", "camelo": "camel", "manzana": "apple-green",
        "cielo": "sky-blue", "miel": "honey", "fuego": "fire-red",
        "ahumado": "smoky grey", "rojizo": "reddish", "offwhite": "off-white",
        "white": "white", "vivo": "bright", "moteado": "mottled",
        "pitaya": "dragonfruit-pink", "altrosa": "dusty pink",
    },
    "lt": {
        "juoda": "black", "balta": "white", "mėlyna": "blue", "raudona": "red",
        "pilka": "grey", "žalia": "green", "rožinė": "pink", "ruda": "brown",
        "geltona": "yellow", "oranžinė": "orange", "smėlio": "sand",
        "kremo": "cream", "auksas": "gold", "aukso": "gold", "azuro": "azure",
        "balkšva": "off-white", "antracito": "anthracite", "alyvinė": "lilac",
        "alyvuogių": "olive", "bazalto": "basalt-grey", "bronzinė": "bronze",
        "dangaus": "sky-blue", "dūmų": "smoky grey", "fuksijų": "fuchsia",
        "gelsvai": "yellowish", "grafito": "graphite", "jūros": "sea-blue",
        "kapučino": "cappuccino", "karamelės": "caramel", "karališka": "royal blue",
        "kario": "curry-yellow", "kivių": "kiwi-green", "konjako": "cognac",
        "kraujo": "blood-red", "kūno": "nude", "lašišų": "salmon",
        "levandų": "lavender", "medaus": "honey", "mėtų": "mint",
        "mišrios": "mixed-color", "margai": "mottled", "nefrito": "jade",
        "obuolių": "apple-green", "omarų": "lobster-red", "opalo": "opal",
        "orchidėjų": "orchid", "persikų": "peach", "pitajų": "dragonfruit-pink",
        "pudros": "powder-pink", "purpurinė": "purple", "rausvai": "pinkish",
        "rusva": "brownish", "rūdžių": "rust", "safyro": "sapphire-blue",
        "sidabrinė": "silver", "smaragdinė": "emerald", "sodri": "rich",
        "šafrano": "saffron", "šokolado": "chocolate-brown", "turkio": "turquoise",
        "ugnies": "fire-red", "uogų": "berry", "vilnos": "wool-grey",
        "vyno": "wine-red", "vyšninė": "cherry-red", "žaliosios": "green",
        "žolės": "grass-green", "žydra": "azure", "citrinos": "lemon-yellow",
        "abrikosų": "apricot", "cerises": "cherry-red", "colors": "colored",
        "nebalintos": "unbleached", "natūrali": "natural", "neoninė": "neon",
        "juodo": "black", "džinso": "denim blue", "rožių": "rose-pink",
        "violetinė": "violet", "perlų": "pearl-white", "nendrių": "reed-beige",
        "geltonumo": "yellow", "geltonosios": "yellow",
    },
    "si": {
        "črna": "black", "bela": "white", "modra": "blue", "rdeča": "red",
        "siva": "grey", "zelena": "green", "roza": "pink", "rjava": "brown",
        "rumena": "yellow", "oranžna": "orange", "bež": "beige", "kaki": "khaki",
        "mornarska": "navy", "moder": "blue", "lila": "lilac", "bordo": "burgundy",
        "antracit": "anthracite", "azur": "azure", "bronasta": "bronze",
        "dimno": "smoky grey", "ecru": "ecru", "gorčica": "mustard",
        "jabolko": "apple-green", "jastog": "lobster-red", "karamel": "caramel",
        "kobalt": "cobalt-blue", "koktejl": "cocktail", "konjak": "cognac",
        "kremna": "cream", "krvavo": "blood-red", "losos": "salmon",
        "majnica": "buttercup-yellow", "marelica": "apricot", "marine": "navy",
        "melona": "melon", "mešane": "mixed-color",
        "off": "off-white", "ognjeno": "fiery red", "olive": "olive",
        "oliva": "olive", "pastelno": "pastel", "pegasto": "spotted",
        "pesek": "sand", "petrol": "petrol-blue", "pitaja": "dragonfruit-pink",
        "puder": "powder-pink", "rjasto": "rust", "rosé": "rose",
        "rubin": "ruby-red", "sepija": "sepia", "smaragd": "emerald",
        "srebrna": "silver", "srebrno": "silver", "staro": "old-rose",
        "turkizna": "turquoise", "večerna": "evening", "vinsko": "wine-red",
        "volneno": "wool-grey", "zabaione": "eggnog-yellow", "zlata": "gold",
        "čokolada": "chocolate-brown", "žafran": "saffron", "progasto": "striped-look",
        "cream": "cream", "black": "black", "denim": "denim blue",
        "travnato": "grass-green", "golobje": "dove-grey", "šampanjec": "champagne",
        "naravno": "natural", "kamela": "camel",
    },
    "lv": {
        "melns": "black", "balts": "white", "zils": "blue", "sarkans": "red",
        "pelēks": "grey", "zaļš": "green", "rozā": "pink", "brūns": "brown",
        "dzeltens": "yellow", "oranžs": "orange", "bēšs": "beige",
        "haki": "khaki", "debeszils": "sky-blue", "antracīta": "anthracite",
        "aprikožu": "apricot", "baložzils": "dove-blue", "bordo": "burgundy",
        "burgundieša": "burgundy", "ceriņu": "lilac", "dabīgi": "natural",
        "dūmu": "smoky grey", "fuksijkrāsas": "fuchsia", "grenadīna": "grenadine-red",
        "jūraszils": "sea-blue", "kamieļkrāsas": "camel",
        "kapučino": "cappuccino", "karameļkrāsas": "caramel", "karija": "curry-yellow",
        "kastaņbrūns": "chestnut-brown", "kastaņkrāsa": "chestnut",
        "kaļķa": "chalk-white", "koraļļu": "coral", "krāsas": "colored",
        "krēmkrāsas": "cream", "lillā": "lilac", "meloņu": "melon",
        "moka": "mocha", "naktszils": "midnight-blue", "nebalināts": "unbleached",
        "nefrīta": "jade", "niedru": "reed-beige", "olīvzaļš": "olive-green",
        "opālisks": "opalescent", "oranžsarkans": "orange-red", "ogu": "berry",
        "olas": "eggshell", "pasteļdzeltens": "pastel yellow",
        "pasteļlillā": "pastel lilac", "pasteļoranžs": "pastel orange",
        "pasteļzaļš": "pastel green", "pelēkbrūns": "grey-brown",
        "persiku": "peach", "piparmētru": "mint", "plūmju": "plum",
        "purpura": "purple", "raibi": "mottled", "rozīgs": "pinkish",
        "rožains": "pink", "rožkrāsas": "pink", "rūsgans": "rusty",
        "sarkanbrūns": "reddish-brown", "sinepjkrāsas": "mustard",
        "smaragda": "emerald", "smilškrāsas": "sand", "sudrabpelēks": "silver-grey",
        "sudrabs": "silver", "vecrozā": "dusty pink",
        "vīnsarkans": "wine-red", "zeltaina": "golden", "ziloņkaula": "ivory",
        "ābolu": "apple-green", "čaumalas": "shell-beige", "šampanieša": "champagne",
        "šokolādes": "chocolate-brown", "ugunssarkans": "fire-red",
        "jauktu": "mixed-color", "krāsu": "colored", "white": "white",
        "denim": "denim blue", "indigo": "indigo", "karaliski": "royal",
        "pērļbalts": "pearl-white", "laša": "salmon", "džinss": "denim blue",
    },
    "ee": {
        # monomorphemic / special color names (not modifier+base compounds)
        "must": "black", "valge": "white", "hall": "grey", "roosa": "pink",
        "sinine": "blue", "roheline": "green", "punane": "red", "pruun": "brown",
        "kollane": "yellow", "oranž": "orange", "lilla": "purple",
        "beež": "beige", "kreem": "cream", "khaki": "khaki", "korall": "coral",
        "oliiv": "olive", "sirel": "lilac", "konjak": "cognac", "kuusk": "spruce-green",
        "lavendel": "lavender", "kuld": "gold", "hõbe": "silver",
        "hõbehall": "silver-grey", "vanaroosa": "dusty pink", "meresinine": "sea-blue",
        "segavärvid": "mixed-color", "teksariie": "denim blue", "denim": "denim blue",
        "kaamel": "camel", "karamell": "caramel",
        "laim": "lime-green", "liiv": "sand", "lõhe": "salmon",
        "melon": "melon", "nu": "nude", "nude": "nude", "opaal": "opal",
        "orhidee": "orchid", "safran": "saffron", "sinep": "mustard",
        "sidrun": "lemon-yellow", "türkiis": "turquoise", "virsik": "peach",
        "õun": "apple-green", "vaarikas": "raspberry-pink", "šoko": "chocolate-brown",
        "kiivi": "kiwi-green", "kitt": "putty-grey",
        "grafiit": "graphite", "jadeiit": "jade", "kani": "cinnamon",
        "kivi": "stone-grey", "loodusvalge": "natural white", "puuder": "powder-pink",
        "petrooleum": "petrol-blue", "täisvalge": "pure white",
        "valkjas": "off-white", "antratsiit": "anthracite", "aprikoos": "apricot",
        "baklažaan": "aubergine", "bordoo": "burgundy", "brokaat": "brocade-gold",
        "burgundia": "burgundy", "cappuccino": "cappuccino", "eosiin": "eosin-pink",
        "elevandiluu": "ivory", "enzian": "gentian-blue", "erkpunane": "bright red",
        "fuksia": "fuchsia", "homaar": "lobster-red", "jõhvikas": "cranberry-red",
        "kahvatulilla": "pale lilac", "karmiinpunane": "crimson",
        "kirsipunane": "cherry-red", "kollakaspunane": "reddish-yellow",
        "koobaltsinine": "cobalt-blue", "lõikega": "cut", "läbipaistev": "transparent",
        "mudavärvid": "mud-color", "munakoor": "eggshell", "mündiroheline": "mint-green",
        "mürk": "toxic-green", "neoonpunane": "neon-red", "neoonroheline": "neon-green",
        "neoonroosa": "neon-pink", "ooker": "ochre", "pastell": "pastel",
        "pastellkollane": "pastel yellow", "pastelloranž": "pastel orange",
        "pastellpunane": "pastel red", "pastellroheline": "pastel green",
        "pastellroosa": "pastel pink", "pastellsinine": "pastel blue",
        "pilliroog": "reed-beige", "ploom": "plum", "punakasvioletne": "reddish-violet",
        "roostepruun": "rust-brown", "roostepunane": "rust-red", "rosé": "rose",
        "rubiinpunane": "ruby-red", "rohekassinine": "teal", "safiir": "sapphire-blue",
        "seemisnahk": "suede-tan", "seepia": "sepia", "smaragdroheline": "emerald-green",
        "suitsuhall": "smoky grey", "suitsusinine": "smoky blue",
        "taevasinine": "sky-blue", "tsüaansinine": "cyan-blue",
        "tulipunane": "fire-red", "tuvisinine": "dove-blue",
        "ultramariinsinine": "ultramarine", "umbra": "umber", "veinipunane": "wine-red",
        "verepunane": "blood-red", "violettsinine": "violet-blue",
        "viimistlemata": "unfinished",
        "öösinine": "midnight-blue", "šampanja": "champagne",
        "helebeež": "light beige", "helehall": "light grey",
        "helekollane": "light yellow", "helelilla": "light purple",
        "heleoranž": "light orange", "helepruun": "light brown",
        "helepunane": "light red", "heleroheline": "light green",
        "heleroosa": "light pink", "helesinine": "light blue",
        "tumebeež": "dark beige", "tumehall": "dark grey",
        "tumekollane": "dark yellow", "tumelilla": "dark purple",
        "tumeoranž": "dark orange", "tumepruun": "dark brown",
        "tumepunane": "dark red", "tumeroheline": "dark green",
        "tumeroosa": "dark pink", "tumesinine": "dark blue",
        "kastanipruun": "chestnut-brown", "kuldkollane": "golden yellow",
        "mariinsinine": "marine blue", "mururoheline": "grass-green",
        "pruunikashall": "brownish grey", "smaragd": "emerald",
        "pitaia": "dragonfruit-pink", "karri": "curry-yellow", "indigo": "indigo",
        "moka": "mocha", "magenta": "magenta", "grenadiin": "grenadine-red",
        "sidrunkollane": "lemon-yellow", "mesi": "honey",
    },
}

# multi-word colour/mixed phrases checked before single-word tokenisation
COLOR_PHRASES = {
    "es": [
        ("mezcla de colores", "multicolor"), ("gris moteado", "mottled grey"),
        ("gris claro", "light grey"), ("gris oscuro", "dark grey"),
        ("azul oscuro", "dark blue"), ("azul claro", "light blue"),
        ("verde oscuro", "dark green"), ("rosa claro", "light pink"),
    ],
    "lt": [
        ("mišrios spalvos", "multicolor"), ("margai pilka", "mottled grey"),
        ("šviesiai mėlyna", "light blue"), ("tamsiai mėlyna", "dark blue"),
        ("šviesiai pilka", "light grey"), ("tamsiai pilka", "dark grey"),
        ("šviesiai žalia", "light green"), ("tamsiai žalia", "dark green"),
        ("šviesiai rožinė", "light pink"), ("tamsiai ruda", "dark brown"),
    ],
    "si": [
        ("mešane barve", "multicolor"), ("svetlo modra", "light blue"),
        ("temno modra", "dark blue"), ("svetlo siva", "light grey"),
        ("temno siva", "dark grey"), ("svetlo zelena", "light green"),
        ("temno zelena", "dark green"), ("svetlo roza", "light pink"),
    ],
    "lv": [
        ("jauktu krāsu", "multicolor"), ("gaiši zils", "light blue"),
        ("tumši zils", "dark blue"), ("gaiši pelēks", "light grey"),
        ("tumši pelēks", "dark grey"), ("gaiši zaļš", "light green"),
        ("tumši zaļš", "dark green"),
    ],
    "ee": [],  # ee compounds modifier+base into a single word already, handled by COLOR_BASE
}

MODIFIER_STANDALONE = {
    "es": {"claro": "light", "oscuro": "dark", "pastel": "pastel"},
    "lt": {
        "šviesiai": "light", "tamsiai": "dark", "tamsi": "dark", "ryškiai": "bright",
        "pastelinė": "pastel", "sodri": "rich", "rusvai": "brownish", "melsvai": "bluish",
        "nakties": "midnight", "rausvai": "pinkish", "gelsvai": "yellowish",
    },
    "si": {"svetlo": "light", "temno": "dark", "pastelno": "pastel", "nočno": "midnight"},
    "lv": {"gaiši": "light", "tumši": "dark"},
    "ee": {"kuninglik": "royal", "meleeritud": "heather-mixed", "meeleritud": "heather-mixed"},
}

STOPWORDS = {
    "de", "a", "s", "y", "and", "at", "in", "by", "of", "the", "des", "le", "la", "z", "o", "on", "for",
    "with", "second", "part", "za", "doma", "na", "med", "spalva", "spalvos", "meta", "kit", "grenada",
    "plažo", "voda", "levi", "yuna", "nebeško", "object", "shirt", "be", "užsegimo", "jdy", "kelnės",
    "glaisto", "brokato", "drobės", "zomšos", "mari", "jan", "luna", "pueblo", "mix", "match", "t",
    "look", "back", "name", "it", "üleriided", "spencer", "lee", "momo", "gandrīz",
    "real", "kennedy", "särk",
}

PATTERN_PHRASE = {
    "solid": "a solid, single-color",
    "floral": "a floral-print",
    "striped": "a striped",
    "animal": "an animal-print",
    "polka_dot": "a polka-dot",
    "mixed": "a mixed-color / heathered",
    "checkered": "a checkered",
    "logo": "a logo-print",
}

LENGTH_PHRASE = {
    "mini": "mini-length (short)",
    "three_quarter": "three-quarter length",
    "knee_midi": "knee-length (midi)",
    "seven_eighth": "seven-eighths length",
    "maxi": "maxi-length (floor-length)",
}


def strip_model(name):
    """Return (name_without_quoted_model, model_or_None).

    The opening quote must be preceded by whitespace/string-start so a brand
    apostrophe glued between letters (LEVI'S, O'NEILL, Marc O'Polo) is never
    mistaken for the start of a quoted style name.
    """
    m = re.search(r"(?:^|(?<=\s))'([^']*)'(?=\s|$)", name)
    if m:
        model = m.group(1).strip()
        rest = name[: m.start()] + " " + name[m.end():]
        return rest, model
    return name, None


def extract_garment(geo, text_lower):
    for substr, en in GARMENT_RULES[geo]:
        if substr in text_lower:
            return en, substr
    return "dress", None


def extract_colors(geo, text_lower):
    """Pull out color phrases/words from text with garment/brand already stripped.
    Returns (ordered unique english color list, leftover unmatched words)."""
    found = []
    remaining = text_lower

    for phrase, en in COLOR_PHRASES.get(geo, []):
        if phrase in remaining:
            found.append(en)
            remaining = remaining.replace(phrase, " ")

    words = re.findall(r"[a-zà-ÿčęėįšųūžāēīūļķņģōõäöüõšž]+", remaining, re.IGNORECASE)
    base = COLOR_BASE[geo]
    modif = MODIFIER_STANDALONE[geo]
    unmatched = []
    pending_modifier = None
    for w in words:
        wl = w.lower()
        if wl in STOPWORDS:
            continue
        if wl in modif:
            pending_modifier = modif[wl]
            continue
        if wl in base:
            c = base[wl]
            if pending_modifier and not any(c.startswith(p) for p in ("light", "dark", "pastel")):
                c = f"{pending_modifier} {c}"
            found.append(c)
            pending_modifier = None
            continue
        pending_modifier = None
        unmatched.append(w)

    seen = set()
    dedup = []
    for c in found:
        if c not in seen:
            seen.add(c)
            dedup.append(c)
    return dedup, unmatched


def extract_brand(text_before_model, text_lower_full, garment_substr, modifier_substrs):
    """Brand = the text before the EARLIEST of the matched garment substring or
    any matched garment-modifier substring (both may appear, and the modifier
    can precede the garment noun, e.g. Estonian 'VERO MODA Kootud kleit')."""
    candidates = []
    if garment_substr:
        candidates.append(text_lower_full.find(garment_substr))
    for msub in modifier_substrs:
        idx = text_lower_full.find(msub)
        if idx != -1:
            candidates.append(idx)
    brand = text_before_model[: min(candidates)] if candidates else text_before_model
    brand = brand.strip(" '\"")
    return re.sub(r"\s+", " ", brand)


def build_description(item):
    geo = item["geo"]
    rest, model = strip_model(item["name"])
    rest_lower = rest.lower()

    garment_en, garment_substr = extract_garment(geo, rest_lower)

    matched_modifiers = []
    for msub, men in GARMENT_MODIFIER.get(geo, []):
        if msub in rest_lower and men not in garment_en:
            garment_en = f"{men} {garment_en}"
            matched_modifiers.append(msub)

    brand = extract_brand(rest, rest_lower, garment_substr, matched_modifiers)

    # strip the matched garment substring (+ any modifier substrings) and the
    # brand out of the remainder before hunting for color words
    color_hunt_text = rest_lower
    if garment_substr:
        color_hunt_text = color_hunt_text.replace(garment_substr, " ")
    for msub in matched_modifiers:
        color_hunt_text = color_hunt_text.replace(msub, " ")
    if brand:
        color_hunt_text = color_hunt_text.replace(brand.lower(), " ", 1)

    colors, unmatched = extract_colors(geo, color_hunt_text)

    subject = f"{brand} {garment_en}" if brand else garment_en
    if model:
        subject += f" (style '{model}')"
    sentence1 = subject
    if colors:
        sentence1 += " in " + " / ".join(colors)
    sentence1 += "."
    sentence2 = f"It has {PATTERN_PHRASE[item['pattern_label']]} pattern and is {LENGTH_PHRASE[item['length_label']]}."
    return f"{sentence1} {sentence2}", unmatched


def main():
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    unmatched_total = 0
    for it in data["items"]:
        desc, unmatched = build_description(it)
        it["description"] = desc
        unmatched_total += len(unmatched)

    MANIFEST.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Added `description` to {len(data['items'])} items in {MANIFEST}")
    print(f"Untranslated leftover word occurrences: {unmatched_total} (dropped silently, never inserted into output)")


if __name__ == "__main__":
    main()
