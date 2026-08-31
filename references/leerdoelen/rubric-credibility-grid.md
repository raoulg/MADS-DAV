# Proposed rubric changes: reward the credibility grid (PTT-54)

`rubric.pdf` is a spreadsheet export, not a file this can edit directly — this is the
concrete text to drop into the source spreadsheet's **Visualisatie: Spurious correlations**
row (5pt group). Everything else in the rubric is unchanged.

## The gap

The current row rewards *"sterke onderbouwing van patronen met statistisch bewijs"* — strong
support, with statistical evidence — for the top tier. Read literally, a student who finds a
significant, unexplained pattern and a student who finds nothing and says so precisely both
have "evidence"; only the first sounds like it satisfies "sterke onderbouwing". Nothing in
the row currently says the second submission should score the same or higher. That is the
exact incentive the *Spurious findings* doc argues against, sitting in the document that
actually assigns the grade.

Three things the row does not currently ask for at all: naming a mechanism (or stating
there isn't one), a replication attempt, and reporting how many comparisons were considered
before landing on the one that's reported.

## Proposed replacement text

**Row: Visualisatie: Spurious correlations**

| column | current | proposed |
|---|---|---|
| Knock outs | *(unchanged)* Het doorlopend missen van spurious correlations, of continu claims maken en dan eindigen met de disclaimer "het zou ook niet zo kunnen zijn" alsof dat een tegengif is. | *add:* Een significant resultaat rapporteren zonder te vermelden hoeveel andere vergelijkingen zijn overwogen. |
| Onvoldoende (0-0.4) | *(unchanged)* | *(unchanged)* |
| Voldoende (0.6-0.8) | Basisaandacht voor validiteit. Enige onderbouwing van patronen. Noemt mogelijke alternatieve verklaringen. | Basisaandacht voor validiteit. Enige onderbouwing van patronen. Noemt mogelijke alternatieve verklaringen. Vermeldt of er een mechanisme is, ook als dat mechanisme ontbreekt. |
| Uitstekend (0.8-1.0) | Sterke onderbouwing van patronen met statistisch bewijs waar dat nodig is om twijfel weg te nemen. Duidelijke afweging van alternatieve verklaringen. Heldere erkenning van beperkingen, zowel statistisch als van het gekozen model. | Voor elke gerapporteerde bevinding: een benoemd mechanisme (of expliciet "geen mechanisme gevonden"), het aantal overwogen vergelijkingen, en — waar mogelijk — een replicatiepoging (andere periode, andere groep, andere maat). **Een bevinding met alle drie, ook als het resultaat "plausibel, onbewezen" is, scoort gelijk aan of hoger dan een significant resultaat zonder mechanisme.** Heldere erkenning van beperkingen, zowel statistisch als van het gekozen model. |
| Specifieke aandachtspunten | *(unchanged)* | *add a line:* Een net zo overtuigend eindresultaat is: "plausibel, onbewezen, en dit is de n die het zou beslissen" — zie leerdoel 5.21 (evidence / mechanism / replication) en de Spurious findings-doc. |

## Also worth a one-line addition

**Row: Reflectie: begeleidende tekst bij elke afbeelding** (3pt) — this is where a student
would actually *write* a "plausible, unproven" verdict, so the Uitstekend column should say
so isn't penalised for lacking a clean significant result:

> *toevoegen aan Uitstekend:* Een verdict als "plausibel, onbewezen" telt hier als een
> heldere, verdedigde conclusie — niet als een ontbrekende conclusie.

## Not proposed here

Leerdoel 5.21 (the three legs: evidence, mechanism, replication) and the checklist.md
"Wat beweer je?" section (PTT-55, already merged) carry the same idea into the teaching
material and the self-check tool. This document only touches the graded rubric, since that
is the piece PTT-54 flags as the actual incentive.

The lesson-1 sentence this issue couldn't wait for M5 — "a null result, honestly bounded, is
a pass" — is already in `notebooks/lesson1/01.2-irc-chat.ipynb`'s closing cell.
