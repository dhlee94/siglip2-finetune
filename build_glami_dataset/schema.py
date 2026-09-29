"""Per-locale field schema for the structured "Label: value, value ... Label2: value"
descriptions used by one particular affiliate feed (looks like About You) across the
GLAMI-1M-dresses geos ee/lt/lv/si/es.

Each geo's description packs many product attributes as consecutive "Label: value"
segments with no delimiter between one field's value and the next field's label
(just a space) - so correct extraction requires knowing the *exact* set of real
field labels for that locale, to correctly find where one field ends and the next
begins. Labels were reverse-engineered empirically (see extract.py's --dump-labels)
by frequency-counting "Capitalized phrase immediately before a colon" across the
manifest and keeping the labels that recur across hundreds/thousands of items.

For "length" (hem length / 기장), several locales reuse the *same* display label for
two different underlying attributes: one is the actual garment length (mini / knee /
maxi ...) and the other is a coarser cut/regularity descriptor (e.g. Estonian
"Tavaline lõige" = "regular cut", Lithuanian "Normalaus ilgio" = "normal length",
Latvian "Garais griezums" = "long cut", Spanish "Normal"). LENGTH_CUT_MARKERS lists
substrings that flag a value as this non-informative "cut" duplicate rather than a
true hem-length value, so it can be filtered out when both appear.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class LocaleSchema:
    geo: str
    # All known field labels for this locale, longest-first so a regex alternation
    # prefers e.g. "Varruka pikkus" over "Pikkus" when both could match.
    boundary_labels: tuple
    pattern_label: str
    length_label: str
    length_cut_markers: tuple = field(default_factory=tuple)


def _sorted_labels(labels):
    return tuple(sorted(set(labels), key=len, reverse=True))


SCHEMAS = {
    "ee": LocaleSchema(
        geo="ee",
        boundary_labels=_sorted_labels([
            "Pikkus", "Istuvus", "Varruka pikkus", "Disain", "Muster", "Lisad",
            "Materjal", "Kaelus", "Üksikasjad", "Jätkusuutlik materjal", "Särgi krae",
            "Kinnitamise tüüp", "Kandmisviis", "Nachhaltigkeit", "Height", "Waist",
            "Imetamisvõimalus", "Bust", "Modelli mõõdud", "Length", "Measurements",
            "Hips", "Material", "Width", "Tootekood", "Rind", "Hip", "Puusad",
            "Girth", "Materiel", "Vöökoht", "Dimensions", "Pakend", "Rinnaga toitmine",
            "Pluusi stiil",
        ]),
        pattern_label="Muster",
        length_label="Pikkus",
        length_cut_markers=("lõige",),
    ),
    "lt": LocaleSchema(
        geo="lt",
        boundary_labels=_sorted_labels([
            "Ilgis", "Pritaikomumas", "Rankovės ilgis", "Raštas", "Papildomai",
            "Dizainas", "Medžiaga", "Detalės", "Iškirptė", "Marškinių apykaklė",
            "Užsegimo tipas", "Pagrindinės medžiagos tipas", "Medžiagos tvarumas",
            "Material", "Sudėtis", "Spalva", "Washing instructions",
            "Kaušelių tipas", "Akinių rėmėlis", "Juosmens aukštis", "Nachhaltigkeit",
            "Gamintojo kodas", "Kolekcija", "Fasonas", "Dydis",
        ]),
        pattern_label="Raštas",
        length_label="Ilgis",
        length_cut_markers=("normalaus ilgio", "ilgio"),
    ),
    "lv": LocaleSchema(
        geo="lv",
        boundary_labels=_sorted_labels([
            "Garums", "Piegriezums", "Piedurkņu garums", "Dizains", "Raksts",
            "Papildinājumi", "Izgriezums", "Materiāls", "Detaļas",
            "Materiāla ilgtspēja", "Slēguma veids", "Krekla apkakle",
            "Valkāšanas veids", "Nachhaltigkeit", "Daļas komplektā",
        ]),
        pattern_label="Raksts",
        length_label="Garums",
        length_cut_markers=("griezums",),
    ),
    "si": LocaleSchema(
        geo="si",
        boundary_labels=_sorted_labels([
            "Dolžina", "Model", "Vzorec", "Dolžina rokavov", "Izrez", "Dizajn",
            "Dodatki", "Material", "Podrobnosti", "Barva", "Kroj", "Ovratnik srajce",
            "Način zapiranja", "Sestava", "Način nošenja", "Trajnost materialov",
            "Rokav", "Zapenjanje", "Kolekcija", "Koda proizvajalca",
            "Funkcija dojenja", "Pranje", "Trajnost", "Podkrilo", "Vzor",
            "Model meri", "Naramnice", "Višina", "Pas", "Boki",
        ]),
        pattern_label="Vzorec",
        length_label="Dolžina",
        length_cut_markers=("cm",),  # prefer the categorical value over a raw cm figure
    ),
    "es": LocaleSchema(
        geo="es",
        boundary_labels=_sorted_labels([
            "Longitud", "Ajuste", "Longitud de la manga", "Estampado", "Diseño",
            "Extras", "Material", "Cuello", "Detalles", "Tipo de cierre",
            "Cuello de camisa", "Sostenibilidad del material", "Tipo de tirante",
            "Para la lactancia", "Import", "Tipo de blusas", "Sostenibilidad",
        ]),
        pattern_label="Estampado",
        length_label="Longitud",
        length_cut_markers=("normal",),
    ),
}
