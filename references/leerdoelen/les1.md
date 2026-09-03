Notebooks: 01.1-goad-toolkit-101, 01.2-irc-chat, 01.3-your-own-chat

De student begrijpt:

1.1 Gestalt principles (proximity, similarity, enclosure, closure, continuity, connection), preattentive processing (de eerste 200ms) en de five guidelines: show the data / reduce clutter / integrate text / avoid spaghetti / start with grey (college en `checklist.md`; in de notebooks vanaf les 2 toegepast)
1.2 dat "wat is één rij?" de eerste vraag van elke analyse is, en dat de vraag bepaalt welke eenheid een rij moet zijn (een dag, een bericht, een persoon)
1.3 dat verrijking — features toevoegen die de vraag nodig heeft en de data niet bevat — het deel van het opschonen is waar de analyse gewonnen wordt
1.4 waarom je na een parse eerst kijkt naar wat er níet matchte, de restcategorieën benoemt en telt, en beslist welke je repareert en welke je laat staan
1.5 de motivatie achter een virtual environment en een projectopzet (`src/`, `data/raw` en `data/processed`, `pyproject.toml`, `notebooks/`, `scripts/`, `tools/`), en kent de vraag die bepaalt waar code hoort: wie anders zou dit nog willen gebruiken (`goad_toolkit` / `wa_analyzer` / `scripts/`)
1.6 dat een null-resultaat, eerlijk begrensd, een voldoende is — en dat een drempel bijstellen tot het verhaal klopt dat niet is

De student kan:

1.7 een virtual environment aanmaken en activeren met `uv`, en de preprocessor vanaf de terminal starten (`click` levert de command-line argumenten)
1.8 regular expressions lezen en schrijven: ankers `^` `$`, shortcuts `\d` `\s` `\S` `\w`, herhaling `+` `*`, groepen `( )`, character classes `[a-z]` en `[^...]`, en het escapen van `[` `]`; weet waarom `\S+` veiliger is dan `.*` en waarom je ankert
1.9 features extraheren met `RegexFeature` in de modes `count` / `has` / `extract`, en weet waarom `extract` zijn eigen dekking logt
1.10 een reeks bewerkingen uitdrukken als een `Pipeline` van benoemde stappen, een eigen stap schrijven door `TransformBase` te subclassen (één methode: `transform`), en `print(pipeline)` lezen als het antwoord op "wat is er met deze data gebeurd"
1.11 de typehints van een `transform`-signatuur lezen als contract
1.12 een stap die af is verplaatsen van een notebook-cel naar `scripts/` en importeren, en beargumenteren waarom een factory (`build_irc_pipeline()`) beter is dan een gedeelde instantie
1.13 bots opsporen op structuur (herhaling, actieve dagen, aandeel op de drukste dag) zonder berichten te lezen, en weet dat herhaling mensen en scripts niet scheidt omdat mensen rituelen hebben
1.14 de eigen chat anonimiseren (`humanize`), de preprocessor draaien en het resultaat via `config.toml` aan `load_own_chat()` koppelen

Python: `re`, pandas `groupby`/`agg`, `itertuples`, `loguru`, goad `Pipeline` / `TransformBase` / `RegexFeature` / `TimeFeatures`, dunder `__call__`

## Vervallen of verplaatst

- ⚠️ regex-onderdelen uit de oude 1.11 die in 01.2 niet meer expliciet geoefend worden: alternatieven `[Bb]`, lookbehind `(?<=...)`, lookahead `(?=...)`. Als het examen ze vraagt is dit een gat.
- ⚠️ `click` (oude 1.9): staat in `src/wa_analyzer/preprocess.py` en de README, wordt in geen notebook uitgelegd.
- "een eigen git-repo maken" (oude 1.10): alleen via `references/01_git crash course.pdf`, niet in het lesmateriaal.
