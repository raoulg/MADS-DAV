# Lesson 4: Distributions, a hypothesis about the process

Exported from the learn app. Edit the goals there, not here.

## De student begrijpt

4.1 dat een verdeling een hypothese is over het proces dat de data maakte, en de drie redenen om er een te fitten in plaats van een gemiddelde te rapporteren: eerlijk samenvatten, bepalen wat ongewoon is, en twee situaties vergelijken
4.2 zes families met het mechanisme erachter — normaal (sommen), lognormaal (producten), exponentieel (wachttijden bij een vaste rate), Poisson (tellingen bij een vaste rate), Weibull (time-to-failure), Pareto (rich-get-richer) — en herkent welk mechanisme bij een variabele in de eigen chat hoort
4.3 het verschil tussen een discrete en een continue verdeling: een telling heeft een kans per waarde (pmf), geen dichtheid (pdf), en `discrete=` is jouw beslissing bij het fitten
4.4 de centrale limietstelling: sommen van bijna alles gaan naar normaal, producten niet — en dat de log een product in een som verandert, waardoor een lange staart een klok wordt
4.5 hoe je een familie fit (maximum likelihood via `DistributionFitter`), waarom er twee winnaars zijn (log-likelihood weegt de bulk, KS de grootste afstand tussen fitted en empirische cdf) en dat hun onenigheid het informatiefste getal in de tabel is
4.6 wat een quantile en een qq-plot zijn en hoe je die leest: de staart, waar families verschillen en histogrammen onleesbaar zijn; en een ECDF als bin-vrije vergelijking van twee samples
4.7 wat een outlier kan zijn — meetfout, zeldzame maar echte waarneming, of teken dat de aangenomen familie niet klopt — en dat "drie standaarddeviaties" een uitspraak is over een normaalverdeling die je niet wist dat je aannam; kan de staartkans onder de gefitte familie berekenen en beargumenteren wat een fout van een zware staart onderscheidt (proceskennis, geen statistiek)
4.8 de belofte van Poisson (variantie = gemiddelde), waarom die in chatdata faalt (de rate zwerft) en dat de negatief-binomiale verdeling daar de familie voor is; kan dezelfde familie vóór en na een bekend event fitten en aflezen welke parameter bewoog (rate, dispersie, schaal)
4.9 hoe een simulatie test of een verschil toeval kan zijn: het label schudden (`NullDistribution`), de wolk plotten en de echte waarde erin plaatsen
4.10 dat wat je kunt zien het effect gedeeld door de spreiding is, dat onzekerheid afneemt met het aantal onafhankelijke waarnemingen, en dat je aan de gewone dagen vooraf kunt aflezen of tien event-dagen de vraag kunnen beslissen
4.11 de modelleerlus: model (vorm + verlies + `train_model`), residu, en een verdeling aan het residu fitten — een symmetrisch residu uit een plausibele familie zonder patroon in de tijd betekent stoppen; een residu met een vorm (stap, drift, bocht) is een vergeten mechanisme, en wannéér het begint is meestal de bevinding

## De student kan

4.13 met `scipy.stats` families maken, samplen en de cdf gebruiken; met `np.log` transformeren
4.14 met goad fitten en tonen: `DistributionRegistry` (en `register_distribution` voor een familie die niet meegeleverd wordt), `DistributionFitter(seed=...)`, `fit` / `fit_distribution`, `fit_table`, `FitResult` / `FailedFit`, `frozen_dist`; `HistogramPlot`, `DistPlot`, `PlotFits`, `QQPlot`, `ECDFPlot`, `NullPlot`
4.15 een pipeline lezen die een tijdreeks klaarmaakt (`DiffValues`, `ShiftValues`, `SelectDataRange`, `RollingAvg`, `ZScaler`) en zeggen welke beslissing elke stap neemt
4.16 een model schrijven als functie van twee inputs en vier parameters, trainen met startwaarden en bounds, en `ResidualPlot` / `ComparePlot(Date)` gebruiken om fit en residu te tonen
4.17 de eigen chat splitsen op een bekend event (uit 03.3), berichten per dag en gaps binnen bursts aan beide kanten fitten, en het resultaat opschrijven als zin met parameters erin
