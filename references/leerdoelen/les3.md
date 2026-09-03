Notebooks: 03.1-time, 03.2-statistics-of-time, 03.3-events-in-your-chat

De student begrijpt:

3.1 dat de dagelijkse en wekelijkse cyclus het saaie, verwachte deel is: niet de bevinding, maar het model dat je aftrekt zodat wat overblijft (het residu) leesbaar wordt
3.2 de afweging van een rolling window: raw en smoothed altijd op één as, en een window zo breed als de cyclus verwijdert de cyclus zelf in plaats van de ruis
3.3 dat een gat in de index niet hetzelfde is als een gat in de plot: continuïteit (gestalt) tekent een lijn dwars door ontbrekende maanden; `reindex` maakt de afwezigheid zichtbaar, en of je daarna invult is een aparte beslissing; weet dat seaborn `NaN`-rijen laat vallen en matplotlib de lijn breekt
3.4 hoe je een bekend event onderzoekt: een eerlijke baseline (release-donderdag tegen gewone donderdagen, niet tegen "alle dagen"), elke dag genormaliseerd naar zijn eigen aandelen zodat één luide dag de pool niet domineert, en het residu (event min gewone dag, per uur) als de plot
3.5 dat een verschil tussen gemiddelden pas iets zegt tegen de spreiding tussen gewone dagen: rijen per dag en een betrouwbaarheidsinterval per balk; tien event-dagen kunnen een verschuiving niet laten zien als de dagen onderling zo veel variëren
3.6 dat een mechanisme gecontroleerd moet worden voor je het opschrijft (de aankondigingstijden), en dat "het trigger is niet bekend" een eerlijker zin is dan een verzonnen reden
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

Python: pandas `.dt`, `resample`, `reindex`, `rolling`, `pivot`; goad `TimeFeatures`, `FlagDates`, `Filter`, `CountValues`, `Share`, `GroupAgg`, `MapValues`, `SortValues`, `SubtractBaseline`, `RollingAvg`; goad `LinePlot`, `FacetPlot`, `HistogramPlot`, `GroupedBarPlot`, `HorizontalLine`, `VerticalDate`, `ACFPlot`, `DecomposePlot`; `FileHandler` voor cachen van downloads; statsmodels `acf`, `seasonal_decompose`; `scipy.fft`; plotly `px.area`

## Vervallen of verplaatst

- oude 3.2 (venv, pdm, path, scripts, git): niet in les 3
- oude 3.8 (linear / sinus / exponential / logistic als basisfuncties): verplaatst naar 04.2 (logistic switch) en 05.3 (sinus op flights) — zie 4.8 en 5.11
- seaborn `FacetGrid` / `.map` (oude 3.10, 3.11): vervangen door goad `FacetPlot`
- ⚠️ `scipy.signal` (oude 3.14): wordt niet meer gebruikt
- ⚠️ materiaal, geen leerdoel: het Fourier-deel van 03.2 (vanaf "3.4 Fourier transforms") is nog in de oude stijl (losse `plt`-plots, uitleg als commentaarregels); de ACF/decompose-helft is wel herschreven
