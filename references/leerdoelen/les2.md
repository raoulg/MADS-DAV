Notebooks: 02.1-baseplot-101, 02.2-comparing_categories, 02.3-simpsons_paradox, 02.4-counting-and-averages, 02.5-your_turn

De student begrijpt:

2.1 de visualisatieprincipes voor een categorische vergelijking: volgorde is een beslissing (sorteer op waarde tenzij de categorie een eigen volgorde heeft), grijs eerst en dan één kleur voor het ding waar de titel over gaat, en een titel is een claim
2.2 twee manieren waarop een groepsvergelijking misleidt terwijl elk getal klopt: te veel ongesorteerde balken in te veel kleuren, en een percentage zonder de aantallen waarop het rust — plus een categorie die stilzwijgend wegvalt omdat hij `NaN` is
2.3 Simpson's paradox: een geaggregeerde vergelijking en elk van de subgroep-vergelijkingen kunnen tegengesteld wijzen zonder rekenfout; de vraag die ertegen beschermt (is er een variabele die verschilt tussen de groepen én de uitkomst beïnvloedt?) en de gebruikelijke verdachten (periode, activiteit, subkanaal, platform, weekdag)
2.4 dat de eenheid van tellen de vraag beantwoordt: berichten tellen en mensen tellen kunnen een tegengesteld antwoord geven, en alleen de eenheid waar de claim over gaat test de claim
2.5 wat een errorbar (95%-betrouwbaarheidsinterval) betekent, dat hij smaller wordt met het aantal onafhankelijke eenheden, en dat minder eenheden betekent dat één afwijkende eenheid het antwoord meer verplaatst
2.6 waarom een `BasePlot`-subclass loont: styling in `PlotSettings`, de claim (data-mapping, titel) bij de aanroep, `build` tekent op `self.ax`, en compositie via `plot_on` / `plot_on_axes` / `create_figure` — ook als argument voor consistentie wanneer een LLM meeschrijft

De student kan:

2.7 een `BasePlot`-subclass schrijven (vijf regels) en een laag die alleen hercolort (`HighlightCategory`-stijl), en de plotklasse verplaatsen naar `scripts/plots.py` zodat een dashboard hem kan importeren
2.8 barplots, grouped bars, heatmaps en barbell plots maken en kiezen wanneer welke past (weinig levels: grouped bar; veel: heatmap; een paar per categorie: barbell)
2.9 small multiples maken met `create_figure(n_plots=...)`, `subplot_titles`, `max_cols` en `sharey`
2.10 een percentage altijd met zijn noemer tonen (twee panelen, of annotatie)
2.11 een geaggregeerde vergelijking splitsen op een kandidaat-verstorende variabele en zoeken naar een omkering, niet alleen een verandering in grootte
2.12 dezelfde vergelijking uitvoeren per bericht en per persoon, met intervallen, en zeggen op hoeveel mensen een claim rust
2.13 de `goad` MCP-tools inzetten als coach: `goad_analysis_checklist` (zes stadia, één tegelijk), `goad_search` / `goad_get_concept`, `goad_critique_visual`

Python: pandas `groupby`/`agg`, `pivot`, `sort_values`, `nlargest`; palettes als dict; goad `PlotSettings` (`highlight`, `base_color`, `highlight_color`, `subplot_titles`, `subplot_ylabels`, `max_cols`, `sharey`), `BarPlot`, `GroupedBarPlot`, `HeatmapPlot`, `BarbellPlot`, `HighlightCategory`, `Annotate`, `ScatterPlot`, `ComparePlotDate`; goad `Pipeline` met `Filter`, `CountValues`, `Head`, `GroupAgg`

## Vervallen of verplaatst

- oude 2.2 / 2.4 (regex oefenen, features extraheren): les 2 importeert lesson 1's pipeline; regex-features komen terug in les 5 (05.1, 05.4) en 03.3
- oude 2.3 (venv, pdm, path, scripts): `pdm` is vervangen door `uv`; venv en paden zijn les 0/1; "code naar scripts/" is hierboven 2.7
- ⚠️ `pandas.cut` (oude 2.8): wordt nergens meer gebruikt
- oude 2.7 (list comprehensions): komt alleen terloops voor, geen leerdoel meer
