# Lesson 5: Relationships, and what makes one believable

Exported from the learn app. Edit the goals there, not here.

## De student begrijpt

5.1 dat een gefitte lijn een claim is over het soort relatie (recht, polynoom, lowess): lowess als beschrijving wanneer je de vorm niet kent, een polynoom als model zodra je die wél kent; en dat een log-log-fit een claim over ratio's is
5.2 waarom je bij vijftien vergelijkingen ongeveer de helft van de tijd iets "significants" vindt als er niets is (1 − 0.95¹⁵), hoe je die basislijn meet door labels te schudden, en waarom gecorreleerde metrics minder dan vijftien kansen zijn (effectief aantal onafhankelijke metrics)
5.3 de verdediging: de vraag opschrijven vóór je kijkt, zeggen hoeveel je bekeken hebt, en het sterkst: de data splitsen en de bevinding op de andere helft laten voorspellen; een bevinding en de zoektocht die hem opleverde zijn één object
5.4 dat de gekozen vergelijking deel is van de claim: een globale baseline kan een echt effect vernietigen (zes van tien) dat een lokale baseline laat zien (negen van tien), omdat een trend een confounder is voor een event
5.5 de drie pijlers van een geloofwaardige bevinding — evidence, mechanism, replication — en het rooster waarin ze gekruist worden; kent de twee off-diagonale uitkomsten: "plausibel, onbewezen" als afgerond resultaat, en een significant resultaat zonder mechanisme als begin van een interessantere vraag; een null-resultaat, eerlijk begrensd, is een voldoende
5.6 dat de eenheid van de toets de eenheid van de claim moet zijn: 425.000 berichten zijn geen 425.000 waarnemingen van mensen; een gepaarde toets per auteur, en het besef dat grote data een kleine n op het niveau van de claim niet repareert (n = 8)
5.7 wat Pearson-correlatie is (covariantie genormaliseerd op de standaarddeviaties) en dat een correlatiematrix niets kan tonen over een variabele die hij nooit kreeg: gepoold versus binnen groepen (Simpson's paradox als scatter), drie manieren om een confounder te hanteren (per groep rapporteren, groepsgemiddelde eraf, in een model), en dat de confounder in je data moet zitten om er iets aan te kunnen doen
5.8 hoe L1-, L2- en elasticnet-regularisatie werken als straf op de gewichten, en waarom de gewichten van een geregulariseerde lineaire regressie een betere feature-selectie zijn dan losse correlaties
5.9 dat een classifier een instrument is, niet het resultaat: de accuracy is een poort, de gewichten zijn de bevinding — en dat je die gewichten leest om te zien of het model gelijk heeft om een reden die generaliseert (typegewoonte) of niet (onderwerp, andermans nicknames, locatie)
5.10 waarom een sterk signaal per bericht toch een lage accuracy geeft (driekwart van de berichten draagt geen vingerafdruk) en dat blokken van berichten de eenheid zijn die de claim nodig heeft; dat stabiliteit over een gat van vier jaar de replicatie-poot is, en dat een maat die níet stabiel is (vragen stellen) de bevinding geloofwaardiger maakt
5.11 het verschil tussen een notebook en een script: kleine functies met expliciete inputs, geen lekkende variabelen, één stap opnieuw draaien, importeren, op een schema draaien; en dat een basisfunctie (linear, sinus, exponential, logistic) een claim is over de vorm die je verwacht, bevestigd doordat het residu krimpt
5.20 dat een basisfunctie een claim is over de verwachte vorm (een logistische schakelaar zegt "de ratio draait"), en dat de parameters een bevinding zijn (de draai zit half maart) terwijl het mechanisme (vaccinatie) een aparte claim blijft die het model niet bewijst

## De student kan

5.12 `RegPlot` gebruiken met `fit_reg`, `order` en `lowess`, met grijze punten en een gekleurde lijn; `scipy.stats.linregress` (ook op `np.log`) en `np.polyfit`
5.13 een Welch-t-toets, gepaarde t-toets en tekentoets kiezen op basis van de eenheid en de vraag, en p-waarden op een log-as plotten met de lijn op 0.05
5.14 een null-basislijn meten door labels te schudden en het aantal "bevindingen" per jacht te plotten
5.15 een correlatieheatmap maken (`CorrelationHeatmap`, pandas `.corr`) en dezelfde correlaties binnen groepen berekenen
5.16 `SGDRegressor` met `penalty="elasticnet"` fitten, de gewichten plotten en interpreteren, en `GridSearchCV` lezen als vooruitblik op hyperparameters
5.17 typegewoontes meten als `RegexFeature`-stappen (smileys met en zonder neus, apostrofs, hoofdletters) en per auteur samenvatten in een heatmap; een `TransformBase`-stap schrijven die berichten tot blokken poolt (`BlockMessages`); twee modellen op dezelfde blokken racen (tf-idf op alle woorden versus tien gewoontes)
5.18 een script met zes functies (`scripts/covid_pipeline.py`) lezen en de functies één voor één in een notebook aanroepen; een extra basisfunctie (sinus met periode 12) aan een lineair model toevoegen met `train_model`
5.19 een bevinding opschrijven zoals hij gerapporteerd hoort te worden: n op het niveau van de claim, de effectgrootte, wat wél en niet getest is, zonder het woord "significant" als conclusie
