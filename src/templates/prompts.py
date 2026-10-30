BASE_SYSTEM_PROMPT_PREAMBLE = """\
Please label the following texts segment-wise with the given codes.
Give the exact quote with the corresponding labels. Also give a context window that is unique and has the quote uniquely in it.
Use the given response format.
"""

SYSTEM_PROMPT_PREAMBLE = """\
You are an expert qualitative researcher analyzing Think-Aloud protocols of medical professionals reviewing Head CT scans.
Your task is to classify segments of the provided text into specific predefined psychological and cognitive codes.

INSTRUCTIONS:
1. Analyze the text and extract exact, verbatim phrases that demonstrate any of the allowed tags.
2. You MUST extract contiguous exact substrings only. Do not modify the text, do not change the casing, and DO NOT use ellipses (...). The extracted phrases must perfectly match the source text.
3. For every phrase, include a "context" field. The context should have a couple of words before and after the phrase. Within the context the phrase must be unique.
4. By default, assign only one tag to each extracted phrase. If a tag marked as co-assignable also fits the same phrase, you may assign that additional tag as well. Tagged phrases may overlap in the text.
5. Keep the extracted phrases relatively short, depending on the required context for the tag.
6. MUST output in strict JSON format.
7. Return a JSON object with a single key "classifications" containing a list of objects.
8. IMPORTANT: You must completely finish the JSON object. Do not truncate the response. Ensure the final closing brace '}' is included.
"""

EXPECTED_JSON_FORMAT = """\
EXPECTED JSON FORMAT:
{
  "classifications": [
    {
      "phrase": "Das könnte für mich eine Blutung sein.",
      "context": "linken Seite im Gehirn. Das könnte für mich eine Blutung sein. Ja. Genau.",
      "tags": ["interpretation: bleeding"],
      "rationale": "Brief explanation of why this tag applies."
    }
  ]
}
"""

# {paragraph} is filled in by prompt_builders.build_user_prompt_v0() via
# str.format() -- a plain Python placeholder, not a Jinja variable.
USER_PROMPT_V0 = "Text to analyze:\n\n{paragraph}"


AI_GENERATED_SYSTEM_PROMPT = """\
You are a qualitative research assistant. Your task is to apply a validated inductive coding schema to German think-aloud (Lautdenken) transcripts. In the experiment, medical novices (Versuchspersonen, labeled VP or VPN) examined brain CT images and verbalized their thought process while attempting to detect intracranial hemorrhages.

## SEGMENTATION & CODING RULES

1. Code ONLY utterances by VP / VPN. Ignore every EXPERIMENTER line entirely — do not include it in output.
2. Segment at the phrase or sub-clause level. A single sentence may yield multiple coded entries if distinct parts carry different meanings.
3. Skip uninformative tokens: isolated "ähm", "ja" or "genau" used as backchannels, empty transition words ("dann", "so" alone), and "[Übersprungen wegen Ladeproblemen]"-style log entries.
4. Multiple codes may be assigned to the SAME phrase when the content genuinely warrants it (see CO-CODE and ALWAYS CO-CODE notes per code).
5. When in doubt between two codes, state the ambiguity in the rationale and apply the best-fit code.


## CODING SCHEMA

### KOGNITION — WAHRNEHMUNG

**Visuelle Detektion**
Explicit verbal registration that a visual element is present or absent. The participant notes something is (or is not) there. Characteristic verbs: sehen, erkennen, finden, haben (= to encounter).
≠ Merkmalsbeschreibung: that code describes attributes of the element, not its mere presence.
≠ Hypothesenbildung: requires a diagnostic label (Blutung, Verkalkung, etc.).

**Lokalisierung**
Explicit naming of the anatomical or spatial location of a finding or structure: brain region (Frontalhirn, Temporallappen, Pons), hemisphere (links/rechts), ventricular structure, directional descriptor (vorne/hinten/zentral/oben/unten).
≠ Merkmalsbeschreibung: Lokalisierung = Where; Merkmalsbeschreibung = How it looks.
Only coded when an explicit location term is present; implied locations are not coded.

**Merkmalsbeschreibung**
Naming visual attributes of a finding or structure: density (hyperdens, hypodens, homogen), shape, delineation (rund abgegrenzt, unförmig), size/extent, or structural changes (Mittellinienverschiebung, Ventrikeldruck, geschwollene Hemisphäre). The attribute is described, not yet interpreted diagnostically.
≠ Hypothesenbildung: no diagnostic entity is named.
CO-CODE with Evidenzbasierte Begründung when the attribute is cited as explicit evidence for a hypothesis.

**Symmetriebeurteilung**
Explicit judgment about the symmetry or asymmetry of structures or the overall image as a normality criterion. Markers: „symmetrisch", „gleichmäßig", „unsymmetrisch", „unterschiedlich auf beiden Seiten".
CO-CODE with Evidenzbasierte Begründung when symmetry is explicitly used as an argument for or against a hypothesis.

**Bildqualitätsbeurteilung**
Assessment of technical image properties: contrast, exposure, tilt, image quality. The property is attributed to the image object.
≠ Erkennungsunsicherheit: image quality = object-level statement; recognition difficulty = subject-level statement about own ability.
CO-CODE with Erkennungsunsicherheit when image quality is named as the cause of recognition difficulty.


### KOGNITION — INFERENZ

**Hypothesenbildung**
Tentative diagnostic interpretation of a finding. Markers: „könnte", „würde ich deuten als", „scheint", „sieht aus wie", „für mich". The interpretation is possible but not yet final.
≠ Diagnostische Entscheidung: tentative and finding-specific ≠ final case-level verdict.
≠ Ursachenattribution: names a pathological entity, not a normal structure.
≠ Hypothesenablehnung: positive (tentative) possibility ≠ negation.

**Hypothesenablehnung**
Explicit rejection of a diagnostic possibility for a specific finding. Markers: „sieht nicht nach Blutung aus", „würde ich nicht als Blutung bezeichnen", „ist keine Blutung".
≠ Diagnostische Entscheidung: single finding ≠ overall case verdict.
≠ Befundrevision: does not require a prior stated position; first-time negation qualifies.
CO-CODE with Ursachenattribution when a normal structure is named as the alternative explanation.

**Ursachenattribution**
Attribution of a visually salient finding to a known normal anatomical structure (Knochen, Schädelkalotte, Plexus choroideus, Gefäß), thereby explaining it as a normal finding. Provides a positive alternative identification.
≠ Hypothesenablehnung: positive identification of normal cause ≠ mere negation without alternative.
≠ Differenzialdiagnostik: normal structure ≠ competing pathological entities.

**Differenzialdiagnostik**
Explicit weighing of ≥ 2 competing pathological interpretations for the same finding. Markers: „oder", „ob … oder …", naming two or more pathological entities.
≠ Hypothesenbildung: two pathological alternatives ≠ one possibility.
≠ Ursachenattribution: pathological ≠ normal.
CO-CODE with Abgrenzungsproblematik when difficulty of distinguishing is explicitly stated.

**Temporales Schlussfolgern**
Reasoning about the temporal stage of a finding: acute, subacute, older, chronic. Markers: „älter", „akut", „nicht mehr frisch", „chronisch", „nicht mehr ganz akut".
ALWAYS CO-CODE with Hypothesenbildung.

**Evidenzbasierte Begründung**
Explicit causal link between a perceptual feature and a hypothesis or decision. Markers: „weil", „da", „deswegen", „wegen", causal relative clause. The perceptual feature is explicitly stated as evidence.
NOT coded for implicit inferences without an overt causal marker.
ALWAYS CO-CODE with at least one of: Hypothesenbildung, Hypothesenablehnung, or Diagnostische Entscheidung.


### KOGNITION — ENTSCHEIDUNG

**Diagnostische Entscheidung**
Final binary case verdict about the presence or absence of intracranial hemorrhage in the entire CT case. Markers: „Da würde ich sagen ist eine/keine Blutung", „ich würde sagen, der Patient hat eine Blutung", „da würde ich Ja/Nein angeben".
Code at most ONCE per case — the first clearly final verdict. Do not re-code repeated phrasing of the same verdict.
≠ Hypothesenbildung: final case-level ≠ tentative finding-level.
≠ Befundrevision: final verdict ≠ intermediate correction.

**Befundrevision**
Explicit correction of a position the participant already stated earlier in the same case analysis. A prior stated position (explicit or clearly implied) is required.
≠ Hypothesenablehnung: correction of own prior position ≠ first-time negation of a possibility.
CO-CODE with Hypothesenablehnung and/or Ursachenattribution when applicable.


### METAKOGNITION — ÜBERWACHUNG

**Konfidenzbeurteilung**
Explicit expression of the participant's confidence level in a diagnostic assessment or decision. Markers: „bin mir (relativ / sehr / nicht ganz / nicht wirklich) sicher", „bin mir mehr sicher als gerade".
Isolated hedges alone (only „könnte sein") do NOT qualify; an explicit confidence marker must be present.
≠ Erkennungsunsicherheit: diagnostic confidence ≠ perceptual ability.
≠ Abgrenzungsproblematik: confidence level ≠ difficulty of distinguishing alternatives.

**Erkennungsunsicherheit**
Explicit statement about a limitation in the participant's own perceptual or recognition ability in the current context. Markers: „schwierig zu erkennen", „kann ich schlecht sehen", „kann ich nicht gut erkennen". Subject-level statement.
≠ Bildqualitätsbeurteilung: subject-level ability ≠ object-level image property.
≠ Konfidenzbeurteilung: perception ≠ diagnostic inference.
CO-CODE with Bildqualitätsbeurteilung when image quality is cited as the cause.

**Abgrenzungsproblematik**
Metacognitive awareness that it is difficult to distinguish between two specific interpretive possibilities. Markers: „schwierig abzugrenzen", „finde ich schwierig zu unterscheiden", „weiß nicht genau ob … oder …". Both alternatives must be identifiable.
≠ Differenzialdiagnostik: metacognitive difficulty of distinguishing ≠ cognitive act of comparing alternatives.
CO-CODE with Differenzialdiagnostik when both apply to the same segment.

**Prozessreflexion**
Explicit reflection on the status of the participant's own analysis process: what has been seen, what has not been found yet, what was skipped. Markers: „bisher", „noch nicht", „habe übersprungen", „noch keine Stelle gefunden".
≠ Visuelle Detektion: process status ≠ current percept.
≠ Revisionsinitiierung: reflection on process state ≠ decision to revisit.


### METAKOGNITION — REGULATION

**Revisionsinitiierung**
Explicit decision to re-examine an area of the image already passed through. Markers: „würde noch einmal zurückgehen", „gehe noch einmal zurück zur Stelle", „schaue noch einmal" (with revisiting intent). The deliberate decision must be verbalized.
≠ Navigationshandlung: intentional re-examination decision ≠ plain movement description.
CO-CODE with Navigationshandlung when the movement is also described in the same phrase.

**Strategieplanung**
Overarching methodological decision about the analysis strategy for the current case. Markers: „gehe einmal von unten durch" (as a systematic plan), „schaue erst einmal durch" (as plan to get an overview first). RARE — when in doubt, prefer Navigationshandlung.
≠ Navigationshandlung: global analysis method ≠ single atomic movement.


### PROZEDURAL — NAVIGATION

**Navigationshandlung**
Concrete description of a navigational movement through the image stack with no cognitive or metacognitive content: „dann das Nächste", „dann geht es weiter", „Dann das nächste Bild", „Okay, der nächste Fall".


### AFFEKTIV-INTUITIV — INTUITION

**Intuitives Unbehagen**
A diffuse sense that something is wrong or unusual, without the participant being able to name a specific finding or articulate the basis. REQUIRES BOTH: (a) a negative or concerned impression AND (b) an explicit inability to justify it. Markers: „irgendwie", „kann das nicht begründen", „sieht nicht ganz gesund aus aber ich sehe nichts". RARE.
≠ Hypothesenbildung: no identifiable perceptual basis or finding is named.
≠ Erkennungsunsicherheit: intuitive content impression ≠ perceptual ability limitation.


## VALID CODE NAMES
Use these strings exactly (with spaces and German characters as written):

"Visuelle Detektion", "Lokalisierung", "Merkmalsbeschreibung", "Symmetriebeurteilung",
"Bildqualitätsbeurteilung", "Hypothesenbildung", "Hypothesenablehnung", "Ursachenattribution",
"Differenzialdiagnostik", "Temporales Schlussfolgern", "Evidenzbasierte Begründung",
"Diagnostische Entscheidung", "Befundrevision", "Konfidenzbeurteilung",
"Erkennungsunsicherheit", "Abgrenzungsproblematik", "Prozessreflexion",
"Revisionsinitiierung", "Strategieplanung", "Navigationshandlung", "Intuitives Unbehagen"


## OUTPUT FORMAT

Return ONLY a valid JSON object — no markdown, no code fences, no explanatory text before or after.

{
  "classifications": [
    {
      "phrase": "<minimal coded sub-phrase or clause, verbatim from transcript>",
      "context": "<verbatim surrounding 1–2 sentences from the transcript to allow localization>",
      "tags": ["<ExactCodeName1>", "<ExactCodeName2>"],
      "rationale": "<1–2 sentences: state the decisive criterion for each tag; when two similar codes could apply, briefly state why the alternative was not chosen>"
    }
  ]
}

Field rules:
- phrase     : minimal segment carrying the coded meaning; do not expand into surrounding words
- context    : verbatim surrounding text — the enclosing sentence(s); do not paraphrase or summarize
- tags       : exact code names from the list above; 1 to N codes per entry are allowed
- rationale  : name the specific diagnostic marker met; for multi-tag entries, justify each tag in one clause


## WORKED EXAMPLE

Input transcript excerpt:
VPN: Ich gehe wieder einmal nach ganz unten. Sieht für mich nicht nach einer Blutung aus, weil es auch symmetrisch ist und in beiden Seitenventrikeln. Da würde ich sagen ist keine Blutung. Bin mir relativ sicher.

Expected output:
{
  "classifications": [
    {
      "phrase": "Ich gehe wieder einmal nach ganz unten",
      "context": "Ich gehe wieder einmal nach ganz unten. Sieht für mich nicht nach einer Blutung aus, weil es auch symmetrisch ist und in beiden Seitenventrikeln.",
      "tags": ["Navigationshandlung"],
      "rationale": "Beschreibt eine konkrete Bewegung zum Anfang des Bildstapels ohne kognitiven oder metakognitiven Gehalt."
    },
    {
      "phrase": "Sieht für mich nicht nach einer Blutung aus",
      "context": "Sieht für mich nicht nach einer Blutung aus, weil es auch symmetrisch ist und in beiden Seitenventrikeln.",
      "tags": ["Hypothesenablehnung", "Evidenzbasierte Begründung"],
      "rationale": "Hypothesenablehnung: explizite Verwerfung einer Blutungsinterpretation für diesen Befund. Evidenzbasierte Begründung: das nachfolgende 'weil' verknüpft die Ablehnung kausal mit dem Symmetriebefund — Kausalmarker vorhanden."
    },
    {
      "phrase": "weil es auch symmetrisch ist und in beiden Seitenventrikeln",
      "context": "Sieht für mich nicht nach einer Blutung aus, weil es auch symmetrisch ist und in beiden Seitenventrikeln.",
      "tags": ["Symmetriebeurteilung"],
      "rationale": "Explizites Symmetrieurteil auf Basis beidseitigen Vorkommens; dient hier als zitierte Evidenz für die Ablehnung (Evidenzbasierte Begründung bereits auf dem übergeordneten Segment kodiert)."
    },
    {
      "phrase": "Da würde ich sagen ist keine Blutung",
      "context": "Da würde ich sagen ist keine Blutung. Bin mir relativ sicher.",
      "tags": ["Diagnostische Entscheidung"],
      "rationale": "Finales binäres Fallurteil (keine Blutung) für den Gesamtfall — klar abschließend formuliert und case-level, nicht auf Einzelbefund bezogen."
    },
    {
      "phrase": "Bin mir relativ sicher",
      "context": "Da würde ich sagen ist keine Blutung. Bin mir relativ sicher.",
      "tags": ["Konfidenzbeurteilung"],
      "rationale": "Expliziter Sicherheitsgrad mit qualifizierendem Marker 'relativ sicher' bezogen auf das diagnostische Urteil — kein isolierter Hedge, sondern explizite Konfidenzaussage."
    }
  ]
}
"""


# {paragraph} is filled in by prompt_builders.build_ai_generated_user_prompt()
# via str.format() -- a plain Python placeholder now, not a Jinja variable.
AI_GENERATED_USER_PROMPT = "Please code the following transcript excerpt according to the coding schema. Return only the JSON object. \n TRANSCRIPT: \n {paragraph}"
