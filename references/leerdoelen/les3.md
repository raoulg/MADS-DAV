De student begrijpt:

3.1 dat "mensen slapen" niet een interessante plot is: het residu ten opzichte van normale verwachte patronen kan dat wel zijn.
3.2 de afweging van een rolling window: raw en smoothed altijd op één as, en een window zo breed als de cyclus verwijdert de cyclus zelf in plaats van de ruis
3.3 dat een gat in de index niet hetzelfde is als een gat in de plot: continuïteit (gestalt) tekent een lijn dwars door ontbrekende maanden; `reindex` maakt de afwezigheid zichtbaar, en of je daarna invult is een aparte beslissing; weet dat seaborn `NaN`-rijen laat vallen en matplotlib de lijn breekt
3.4 hoe je een bekend event onderzoekt: een eerlijke baseline (release-donderdag tegen gewone donderdagen, niet tegen "alle dagen"), elke dag genormaliseerd naar zijn eigen aandelen zodat één luide dag de pool niet domineert, en het residu (event min gewone dag, per uur) als de plot
3.5 dat een verschil tussen gemiddelden pas iets zegt tegen de spreiding tussen gewone dagen: rijen per dag en een betrouwbaarheidsinterval per balk; tien event-dagen kunnen een verschuiving niet laten zien als de dagen onderling zo veel variëren
3.6 dat een mechanisme gecontroleerd moet worden voor je het opschrijft (de aankondigingstijden), en dat "de trigger is niet bekend" een eerlijker zin is dan een verzonnen reden
3.7 dat een ratio niets verklaart: de vorm van de dag, per periode genormaliseerd, en een controlekanaal maken een verhaal ("chatten vanaf werk") geloofwaardig
3.8 autocorrelatie als manier om een onbekende periode te meten, seasonal decompose (trend, seizoen, residu) met die gemeten periode, en dat je het residu controleert of het afgetrokken deel echt weg is — bij zonnevlekken is dat niet zo, omdat de cycli niet even lang zijn
3.9 het globale idee van een Fourier-transformatie (elk patroon als som van sinussen) en waarom meer componenten overfitten
3.10 dat je in de eigen chat begint bij een event dat je kent, de verwachte impact opschrijft vóór je kijkt (hoeveel / wanneer / wie / hoe), en dan de ene plot bouwt die het zou laten zien — niet andersom

De student kan:

3.11 met timestamps werken: `pd.to_datetime`, `.dt`-accessors, `resample("D")`, `date_range` + `reindex`, `rolling`
3.12 een event-study bouwen als pipeline: `FlagDates` → `Filter` → `CountValues(normalize=True)` / `GroupAgg` + `Share(by=...)` → `SubtractBaseline`, en een eigen `TransformBase`-stap schrijven voor het model (`MeanOfDayShares`)
3.13 een geïmporteerde pipeline uitbreiden met `.add(...)` (`BuildTimestamp`, `TimeFeatures`)
3.14 small multiples per weekdag of per auteur maken met `FacetPlot`, en per-dag rijen met `errorbar=("ci", 95)` plotten
3.15 een ACF plotten en de sterkste lag na de eerste stretch aflezen; `seasonal_decompose` draaien en het residu opnieuw op autocorrelatie testen

Python: 
- pandas `.dt`, `resample`, `reindex`, `rolling`, `pivot`; 
- goad `TimeFeatures`, `FlagDates`, `Filter`, `CountValues`, `Share`, `GroupAgg`, `MapValues`, `SortValues`, `SubtractBaseline`, `RollingAvg`; 
- goad `LinePlot`, `FacetPlot`, `HistogramPlot`, `GroupedBarPlot`, `HorizontalLine`, `VerticalDate`, `ACFPlot`, `DecomposePlot`;
-  `FileHandler` voor cachen van downloads; 
- statsmodels `acf`, `seasonal_decompose`; 
- `scipy.fft`; 
- plotly `px.area`
