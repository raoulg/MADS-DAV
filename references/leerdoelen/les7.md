# Lesson 7: Networks, and what a cluster is worth

Exported from the learn app. Edit the goals there, not here.

## De student begrijpt

7.1 dat een graaf nodes en edges is en dat de edge-regel de analyse ís: dezelfde berichten geven onder een andere regel (wie was in de buurt / wie werd aangesproken) een ander netwerk, een andere centrale persoon en een andere zin in het rapport — en dat een gulle regel prima is zolang je zegt wat hij niet kan onderscheiden
7.2 dat het tijdvenster en de definitie van een node (nick-varianten, een bot die altijd aanwezig is) beslissingen zijn die in het bijschrift horen, en dat elke maat op de graaf die keuzes erft: een centraliteit is een claim over een regel, niet over een persoon
7.3 wat degree, weighted degree, betweenness en clustering coefficient meten, dat het verschillende prestaties zijn (de Medici zijn hoogste in beide, Strozzi en Guadagni delen een degree en verschillen een factor 2,5 in betweenness), en dat een centraliteit een rangorde binnen één graaf is en geen grootheid daarbuiten
7.4 dat positie in een layout geen data is (zeg de degree, niet "aan de rand"), en dat een maat die sterk correleert met een kolom die je al had — of een "gedeeld vocabulaire" dat berichten telt — een omweg is en geen vondst; die check kost één regel en hoort vóór de interpretatie
7.5 wat de Laplaciaan $L = D - A$ en het spectrum over de structuur van een graaf zeggen: het aantal nul-eigenwaarden is het aantal componenten, de Fiedler-vector snijdt, en de eigengap is het enige wat de graaf zelf over $k$ zegt
7.6 dat modulariteit pas een claim wordt naast een null van herbedrade grafen, een resolutielimiet heeft, en dat clusters pas iets zijn als ze een storing overleven (seed, resolutie, drempel, een andere helft van het jaar, een andere edge-regel) — en dat een scherp begrensd nulresultaat een volwaardig resultaat is, terwijl een graaf die een klok meet geen sociaal netwerk is
7.7 waarom een `BasePlot`-subclass uit les 2 ongewijzigd in een dashboard rendert, en dat elke slider in een dashboard een beslissing is die je aan de lezer overlaat — het venster als slider is precies de parameter die je moet kunnen voelen

## De student kan

7.8 twee edge-regels bouwen op dezelfde berichten (`GraphBuilder` met `Config`, `networkx` uit een `groupby` op `addressed_to`), ze als verzamelingen edges vergelijken en de degrees per persoon tegen elkaar uitzetten, en een venster-sweep draaien met een tweede kanaal als replicatie
7.9 een graaf filteren, een layout berekenen en tonen (`GraphVisualizer`, `GraphPlot` met twee seeds naast elkaar zodat elk getal identiek blijft), en een streamlit-dashboard lezen en aanpassen (`st.selectbox`, `st.session_state`, `st.slider`, `@st.cache_data`, `st.plotly_chart`)
7.10 degree, weighted degree, betweenness en clustering coefficient met `networkx` berekenen op de giant component, ze onderling en tegen een bestaande kolom correleren met `spearmanr`, en componenten, geïsoleerde nodes en de weggevallen fractie rapporteren
7.11 clusters vinden en toetsen: spectraal (adjacency en Laplaciaan, `eigh`, eigengap, Fiedler-vector, `KMeans` op de eerste $k$ eigenvectoren) en met `louvain_communities`, Q rapporteren naast een null van herbedrade grafen, twee clusteringen vergelijken met `adjusted_rand_score`, en clusters valideren tegen kolommen die het algoritme nooit zag (`kruskal`, uur-profiel als heatmap)
7.12 op de eigen chat de hele keten draaien: de graaf op de nabijheidsregel bij twee vensters, Q met null en z, een split in twee helften met de ARI ertussen, een validatie tegen een ongebruikte kolom — en de edge-regel als zin opschrijven ("twee mensen zijn verbonden als …", inclusief venster), wat die regel niet kan onderscheiden, één getal dat geen plaatje is, en wat er met de nodes gedaan is
