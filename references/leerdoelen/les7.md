Notebook: 07.1-social_graphs. Dashboards: `dashboards/dashboard_1.py` t/m `dashboard_5.py` en `streamlit_app.py`

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

Python: `networkx` (`Graph`, `add_edge(weight=)`, `degree`, `edges`), `frozenset` voor ongerichte edges, `wa_analyzer.network_analysis` (`Config`, `GraphBuilder`, `GraphVisualizer`), pandas `rank` / `corr(method="spearman")`, goad `ScatterPlot` + `ax.annotate`, `scripts.plots.BarPlot`, streamlit

## Vervallen of verplaatst

- de oude 7.1–7.6 (streamlit: selectbox, session_state, multiselect/radio, if-elif-else, columns, model in session_state) zijn allemaal nog gedekt, maar alleen in `dashboards/` en de README daar — ⚠️ er is geen notebook dat streamlit stap voor stap doorloopt; 07.1 verwijst ernaar in 7.1.6. Beslissen of de dashboards als zelfstudie volstaan of dat er een 07.2 bij hoort.
- de slides van les 7 (DAV_7) gaan over storytelling en de rubric, niet over graven of streamlit
