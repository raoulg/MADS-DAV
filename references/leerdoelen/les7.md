Notebooks: 07.1-social_graphs, 07.2-graph-properties, 07.3-clusters-in-graphs. Dashboards: `dashboards/dashboard_1.py` t/m `dashboard_5.py` en `streamlit_app.py`. Naslag: Barabási, *Network Science* (http://networksciencebook.com/), hoofdstuk 2 (graaftheorie) en 9 (communities)

De student begrijpt:

7.1 dat een graaf nodes en edges is, dat "praatte met" geen kolom in de data is, en dat de definitie van een edge de analyse ís: dezelfde berichten geven onder twee regels (wie was in de buurt / wie werd aangesproken) een ander netwerk, een andere centrale persoon en een andere zin in het rapport
7.2 dat de nabijheidsregel gul is: hij vindt bijna elk aangesproken paar (97%) plus twee derde edges zonder aangesproken tegenhanger, en kan die niet van elkaar onderscheiden — prima zolang je het zegt, fataal zodra je "deze twee zijn close" schrijft over één paar
7.3 dat een regex-feature (`addressed_to`) geen nick-detector is: een derde van de matches zijn woorden als "yeah," en de check `isin(nicks)` kost één regel — maar alleen als je weet dat je hem moet doen
7.4 dat het tijdvenster een draaiknop op de claim is, geen weergave-optie: het hoort in het bijschrift, en een tweede kanaal is de check dat de vorm van het antwoord standhoudt
7.5 dat nodes ook een beslissing zijn (nick-varianten, een bot met 42 buren en één aangesprokene) en dat een centraliteitsscore een claim is over een regel, niet over een persoon
7.6 wat een spring layout doet, en dat positie op het scherm geen data is: zeg de degree, niet "aan de rand van het netwerk"; en dat degree in een kleine groepschat weinig zegt, edge weight meer
7.7 waarom een `BasePlot`-subclass uit les 2 ongewijzigd in een dashboard rendert (`plot()` geeft een matplotlib-figuur, `st.pyplot` neemt er een), en dat elke slider in een dashboard een beslissing is die je aan de lezer overlaat — het venster als slider is precies de parameter die je moet kunnen voelen

De student kan:

7.8 twee edge-regels bouwen op dezelfde berichten (`GraphBuilder` met `Config`, en `networkx` uit een `groupby` op `addressed_to`), ze als verzamelingen edges vergelijken (gevonden / zonder tegenhanger) en de degrees per persoon tegen elkaar uitzetten
7.9 een venster-sweep draaien en de twee getallen samen laten bewegen; een tweede kanaal als replicatie toevoegen
7.10 het netwerk filteren (`filter_connections`), een layout berekenen en met `GraphVisualizer` interactief tonen (plotly)
7.11 de eigen chat als graaf bouwen op de nabijheidsregel, bij twee vensters, en één paar benoemen dat de graaf verbindt terwijl je weet dat het niet klopt
7.12 een streamlit-dashboard lezen en aanpassen: `st.selectbox` met `st.pyplot` (dashboard 1), `st.session_state` voor data en een getraind model (1, 4), `st.multiselect` en `st.radio` met if/elif/else-logica (2), `st.columns` en een filter die elke chart volgt (3), `st.button` met een voorspelling (4), `st.slider`, `@st.cache_data` en `st.plotly_chart` (5)
7.13 de edge-regel als zin opschrijven ("twee mensen zijn verbonden als …", inclusief venster), wat die regel niet kan onderscheiden, één getal dat geen plaatje is, en wat er met de nodes gedaan is

Over grafen meten en clusteren — de student begrijpt:

7.14 wat degree, betweenness en clustering coefficient betekenen op een graaf waar het antwoord van buiten de data bekend is (de Florentijnse families): de Medici hebben zowel de hoogste degree (6) als de hoogste betweenness (0.52) en dat zijn twee verschillende prestaties; Strozzi en Guadagni hebben dezelfde degree en een factor 2,5 verschil in betweenness, omdat de één in een clique zit en de ander tussen twee delen die geen andere route hebben
7.15 dat elke maat de edge-regel erft en dus een claim over die regel is: dezelfde mensen krijgen onder de nabijheidsregel en de aanspreekregel een clustering coefficient die tegengesteld met degree correleert (−0.59 en +0.46), en beide zinnen ("hubs zitten tussen groepen" / "hubs zitten ín groepen") zijn verdedigbaar zolang de regel erbij staat
7.16 dat een centraliteit die 0.92 correleert met een kolom die je al had (weighted degree tegen het aantal berichten) geen vondst is maar een omweg — en dat die check één regel kost en vóór de interpretatie hoort
7.17 dat een bot een regelprobleem is en geen schoonmaakprobleem: `lubotu3\`` koopt met 36 berichten de 36e degree van 404, omdat de nabijheidsregel aanwezigheid meet en een script altijd aanwezig is; wie op `bot` grept vindt de mens met een altijd openstaande client niet
7.18 dat een centraliteit een rangorde binnen één graaf is en geen grootheid daarbuiten (genormaliseerde betweenness 0.52 bij n=15 tegen 0.18 bij n=404, ruw precies andersom en een factor 300), en dat het aantal componenten en geïsoleerde nodes in het bijschrift hoort (de aanspreekgraaf is 188 stukken, waarvan 187 losse mensen)
7.19 wat de Laplaciaan $L = D - A$ is, waarom juist de kléínste eigenwaarden tellen ($x^{\top}Lx$ is de som van kwadratische onenigheid over de edges), wat de Fiedler-vector is, dat het aantal nul-eigenwaarden het aantal componenten is, en dat de eigengap het enige is wat de graaf zelf over $k$ zegt
7.20 dat modulariteit geen score is maar een verschil: elke graaf heeft een Q, ook een graaf zonder structuur, en pas naast de Q van twintig herbedrade grafen met dezelfde degree-sequentie wordt het een claim — karate Q=0.42 met z=+11 tegen de Florentijnse families Q=0.40 met z=+0.6, waar de hógere Q de betekenisloze is
7.21 dat modulariteit een resolutielimiet heeft die de karateclub in vier stukken hakt in plaats van twee, en dat de resolutie terugdraaien tot 0.6 alleen "werkt" omdat het antwoord al bekend was
7.22 dat clusters pas iets zijn als ze een storing overleven — seed, resolutie, drempel, een andere helft van het jaar, een andere edge-regel — en dat een verschoven drempel die de helft van de mensen weglaat geen storing is maar een andere vraag
7.23 dat een nulresultaat, scherp begrensd, een volwaardig resultaat is: de aanspreekgraaf van #ubuntu-uk heeft geen verdedigbare communities (z=+2.6, ARI 0.08 tussen de twee helften van 2015, ARI 0.18 tegen de nabijheidsregel, en alleen het uur van de dag onderscheidt de clusters, met 14% van de variantie)
7.24 dat de graaf die wél sterk clustert (uur-profielen, Q=0.48 en z=+46, ook ná normaliseren per uur) een klok meet en geen sociaal netwerk: hij is het oneens met de aanspreekgraaf (ARI 0.18), en niemand in het 22:00-cluster hoeft ooit een woord met een ander lid gewisseld te hebben
7.25 dat een edge-regel die als "gedeeld vocabulaire" wordt gepresenteerd een activiteitsmaat kan zijn: de gemiddelde cosinus tussen tf-idf-profielen correleert 0.96 met het aantal berichten, want een lang document dekt meer vocabulaire — één regel, vóór het clusteren, en de hele constructie is afgewezen

Over grafen meten en clusteren — de student kan:

7.26 degree, weighted degree, betweenness en clustering coefficient met `networkx` berekenen op de giant component, ze onderling en tegen een bestaande kolom correleren met `spearmanr`, en componenten, geïsoleerde nodes en de weggevallen fractie rapporteren
7.27 dezelfde graaf met twee spring-layout-seeds naast elkaar tekenen (`GraphPlot`) en laten zien dat elk getal identiek blijft
7.28 de adjacency- en Laplaciaan-matrix opbouwen, `np.linalg.eigh` draaien, het spectrum plotten, de eigengap aflezen, nodes kleuren op het teken van de Fiedler-vector, `KMeans` op de eerste $k$ eigenvectoren draaien, en overstappen op de genormaliseerde Laplaciaan wanneer de ongenormaliseerde alleen degree-1-nodes afknipt
7.29 `louvain_communities` draaien, Q rapporteren náást een null van herbedrade grafen (`modularity_check` met `double_edge_swap`), en twee clusteringen vergelijken met `adjusted_rand_score` over de nodes die ze delen
7.30 clusters valideren tegen kolommen die het algoritme nooit zag (aantal berichten, mediaan uur, url-aandeel, vraagtekens) met `kruskal` en het aandeel verklaarde variantie, en het uur-profiel per cluster als heatmap tonen
7.31 een edge-regel bouwen uit een profiel in plaats van uit contact (`HourProfile`, `cosine_edges`) en de normalisatie per uur gebruiken als tóets op het bezwaar dat iedereen 's nachts slaapt
7.32 op de eigen chat de hele keten draaien: Q met null en z, een split van de data in twee helften met de ARI ertussen, een validatie tegen een ongebruikte kolom, en de zin opschrijven die die getallen precies dragen

Python: `networkx` (`Graph`, `add_edge(weight=)`, `degree`, `edges`, `betweenness_centrality`, `clustering`, `average_clustering`, `diameter`, `average_shortest_path_length`, `connected_components`, `to_numpy_array`, `double_edge_swap`, `spring_layout`, `community.louvain_communities` / `community.modularity`, `florentine_families_graph`, `karate_club_graph`), `frozenset` voor ongerichte edges, `numpy.linalg.eigh`, sklearn `KMeans` / `adjusted_rand_score` / `TfidfVectorizer`, scipy `spearmanr` / `kruskal`, `wa_analyzer.network_analysis` (`Config`, `GraphBuilder`, `GraphVisualizer`), `scripts.graphs` (`MentionEdges` en `HourProfile` als `TransformBase`, `cosine_edges`, `to_graph`, `giant_component`, `best_partition`, `modularity_check`, `labels`, `GraphPlot` als `BasePlot`), `scripts.pipelines.BuildTimestamp`, pandas `rank` / `corr(method="spearman")`, goad `ScatterPlot` + `ax.annotate` / `HeatmapPlot` / `LinePlot`, `scripts.plots.BarPlot`, streamlit
