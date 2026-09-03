Notebooks: 06.1-dimensionality_reduction, 06.2-modelling, 06.3-vectorspaces, 06.4-your-own-vectors

De student begrijpt:

6.1 de motivatie voor het embedden van data in een hoogdimensionale vectorruimte: een representatie waarin "dichtbij" "lijkt op" betekent, zodat je erin kunt zoeken, clusteren en een eigen model op kunt trainen; en dat het embedden losstaat van de analyse (een cache is vectoren plus een manifest, één keer berekend en gepubliceerd)
6.2 wat PCA zoekt: de lijn met de meeste spreiding erlangs is precies de lijn waar de punten het dichtst bij liggen (Pythagoras), en dus de projectie die het minst verliest; de drie delen van een SVD (richtingen, singuliere waarden, posities) en de betekenis van orthogonaal, genormaliseerd en basis
6.3 dat variantie eenheden heeft: zonder standaardiseren vindt PCA de kolom met de grootste getallen (grammen), en een component van 99,99% is dan het symptoom, niet de triomf; wanneer je níet standaardiseert (alle kolommen dezelfde eenheid, pixels)
6.4 dat een projectie altijd zijn kader vult, en dat de scree plot (aandeel verklaarde variantie) daarom in het bijschrift hoort
6.5 het verschil tussen PCA en t-SNE: PCA is een platte schaduw die de globale geometrie bewaart en een kromming niet kan ontrollen; t-SNE bewaart buren en ontrolt, maar geeft de globale geometrie op (de swiss roll); en de manifold-hypothese die het gebruik van t-SNE op 784 pixels rechtvaardigt
6.6 dat perplexity geen weergave-optie is: lidmaatschap van clusters overleeft een sweep, de geometrie (afstanden, groottes, gaten) niet; wat je wél en níet mag zeggen over een t-SNE-plot; dat pure ruis bij lage perplexity clusters oplevert en dat je daarom nooit de embedding clustert maar de data
6.7 hoe tekst een vector wordt door te tellen: character-trigrammen met `CountVectorizer`, waarom de chunks even groot moeten zijn, en de Manhattan-afstand als som van absolute telverschillen; dat een telvector terug te lezen is (welke trigrammen doen het werk) en dat platform-conventies (links, quote-markup) en schrijfstijl daarin door elkaar lopen
6.8 het verschil tussen cosinus-similariteit en Euclidische / Manhattan-afstand: cosinus negeert lengte, en of dat uitmaakt controleer je aan de normen van je vectoren (een encoder die zelf normaliseert maakt de keuze onbelangrijk; een WhatsApp-bericht niet)
6.9 dat een 1-nearest-neighbour-baseline zonder training het getal is dat elke embedding-methode moet verslaan, en dat een getal in de oorspronkelijke ruimte controleert wat een 2D-plaatje alleen suggereert
6.10 dat "wat is één rij" ook voor tekst de eerste vraag is: losse berichten embedden tot een klomp zonder structuur; sessies (drempel gefit met de exponentiële verdeling uit les 4) zijn de eenheid die betekenis draagt, en een self-supervised check (de ene helft zoekt de andere) toont welke samenvoeging werkt
6.11 dat een semantische embedding meet waarvoor hij getraind is — inhoud, niet gewoonte — en dat trigram-tellingen daardoor op auteurschap winnen van een sentence-embedding; kies het instrument bij de vraag, en race ze voor je kiest
6.12 de curse of dimensionality voor zover de ruis-demo hem laat zien: in vijftig dimensies heeft elk punt buren, ook als er niets te vinden is

De student kan:

6.13 `np.linalg.svd` en `sklearn` `PCA` toepassen, controleren dat ze hetzelfde geven (op een teken na), en `explained_variance_ratio_` met `ScreePlot` tonen
6.14 `StandardScaler` toepassen vóór PCA en de loadings per component lezen als zin ("PC1 is grootte")
6.15 `TSNE` draaien met meerdere perplexities, `ProjectionPlot` in een grid gebruiken, en silhouettes (naar echte labels, en met geschudde labels) berekenen als getal zonder plaatje
6.16 een tekstcorpus opschonen en in chunks knippen als `TransformBase`-stappen (`NormaliseText`, `ChunkByCharacters`), trigrammen tellen, Manhattan-afstanden berekenen, projecteren, en dezelfde stappen op de eigen chat per auteur draaien
6.17 een gepubliceerde stijlometrie-claim ("twee auteurs") langs het rooster van les 5 leggen: het bewijs benoemen, twee mechanismen die het plaatje allebei voorspellen, en opschrijven wat het zou beslissen
6.18 voorberekende vectoren en labels van de hub laden (`VectorCache.from_hub`), cosinus als dot-product na normaliseren berekenen, en met torch een 1-NN-baseline schrijven
6.19 met `sentence-transformers` embedden, `scripts.sessionize` (`fit_session_threshold`, `sessionize`, `merge_messages`) gebruiken, een self-retrieval-check schrijven, en een `transformers.pipeline` sentiment-model als korte afsluiter draaien (tabel, geen claim)

Python: `numpy.linalg.svd`, sklearn `PCA` / `TSNE` / `StandardScaler` / `silhouette_score` / `KMeans` / `CountVectorizer` / `manhattan_distances` / `LogisticRegression`; `fetch_openml`, `make_swiss_roll`; torch `normalize` / `cdist`; `vectormesh.VectorCache`; `SentenceTransformer.encode`; `transformers.pipeline`; goad `ProjectionPlot`, `ScreePlot`, `HeatmapPlot`, `FileHandler`; `notebooktester.param` voor snelle CI-runs

## Vervallen of verplaatst

- ⚠️ de analogie van een vectorruimte met het opruimen van een kamer (oude 6.2): staat in geen notebook en niet in de slides
- ⚠️ de definitie van een vectorruimte passief kunnen volgen (oude 6.5): niet meer in het materiaal; de metric-helft (oude 6.6) zit in 6.8
- ⚠️ curse of dimensionality (oude 6.3): alleen via de ruis-demo (6.12); de "grote getallen"-intro staat in de slides
- `.apply()` met `pd.to_datetime` (oude 6.15): de tijd-bins van het oude 06.2 zijn vervallen (ze werden nergens gebruikt); `pd.to_datetime(..., unit="s")` komt nog wel voor
- `@dataclass` (oude 6.16): verplaatst naar les 0 (00.1-classes)
- de sentiment-pipeline (oude 6.17) is een korte afsluiter in 06.4, geen leerdoel op zichzelf
