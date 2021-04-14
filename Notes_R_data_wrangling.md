# Learn R (youtube video: R programming tut)



## R studio

```R
# File:   RStudio.R
# Course: R: An Introduction (with RStudio)

# LOAD DATA ################################################

library(datasets)  # Load built-in datasets

# SUMMARIZE DATA ###########################################

head(iris)         # Show the first six lines of iris data
summary(iris)      # Summary statistics for iris data
plot(iris)         # Scatterplot matrix for iris data

# CLEAN UP #################################################

# Clear packages
detach("package:datasets", unload = TRUE)  # For base

# Clear plots
dev.off()  # But only if there IS a plot

# Clear console
cat("\014")  # ctrl+L

# Clear mind :)

```

## Packages

where to get packages

CRAN	

cranstastic

github

```R
# File:   Packages.R
# Course: R: An Introduction (with RStudio)

# LOAD PACKAGES ############################################

# I recommend "pacman" for managing add-on packages. It will
# install packages, if needed, and then load the packages.
install.packages("pacman")

# Then load the package by using either of the following:
require(pacman)  # Gives a confirmation message.
library(pacman)  # No message.

# Or, by using "pacman::p_load" you can use the p_load
# function from pacman without actually loading pacman.
# These are packages I load every time.
pacman::p_load(pacman, dplyr, GGally, ggplot2, ggthemes, 
  ggvis, httr, lubridate, plotly, rio, rmarkdown, shiny, 
  stringr, tidyr) 

library(datasets)  # Load/unload base packages manually

# CLEAN UP #################################################

# Clear packages
p_unload(dplyr, tidyr, stringr) # Clear specific packages
p_unload(all)  # Easier: clears all add-ons
detach("package:datasets", unload = TRUE)  # For base

# Clear console
cat("\014")  # ctrl+L

# Clear mind :)

```

## Plot()

```R
# File:   Plot.R
# Course: R: An Introduction (with RStudio)

# LOAD DATASETS PACKAGES ###################################

library(datasets)  # Load/unload base packages manually

# LOAD DATA ################################################

head(iris)

# PLOT DATA WITH PLOT() ####################################

?plot  # Help for plot()

plot(iris$Species)  # Categorical variable
plot(iris$Petal.Length)  # Quantitative variable
plot(iris$Species, iris$Petal.Width)  # Cat x quant box
plot(iris$Petal.Length, iris$Petal.Width)  # Quant pair
plot(iris)  # Entire data frame

# Plot with options
plot(iris$Petal.Length, iris$Petal.Width,
  col = "#cc0000",  # Hex code for datalab.cc red
  pch = 19,         # Use solid circles for points
  main = "Iris: Petal Length vs. Petal Width",
  xlab = "Petal Length",
  ylab = "Petal Width")

# PLOT FORMULAS WITH PLOT() ################################

plot(cos, 0, 2*pi)
plot(exp, 1, 5)
plot(dnorm, -3, +3)

# Formula plot with options
plot(dnorm, -3, +3,
  col = "#cc0000",
  lwd = 5,
  main = "Standard Normal Distribution",
  xlab = "z-scores",
  ylab = "Density")

# CLEAN UP #################################################

# Clear packages
detach("package:datasets", unload = TRUE)

# Clear plots
dev.off()  # But only if there IS a plot

# Clear console
cat("\014")  # ctrl+L

# Clear mind :)

```

## Bar Charts

```R
# File:   BarCharts.R
# Course: R: An Introduction (with RStudio)

# LOAD DATASETS PACKAGES ###################################

library(datasets)

# LOAD DATA ################################################

?mtcars
head(mtcars)

# BAR CHARTS ###############################################

barplot(mtcars$cyl)             # Doesn't work

# Need a table with frequencies for each category
cylinders <- table(mtcars$cyl)  # Create table
barplot(cylinders)              # Bar chart
plot(cylinders)                 # Default X-Y plot (lines)

# CLEAN UP #################################################

# Clear environment
rm(list = ls()) 

# Clear packages
detach("package:datasets", unload = TRUE)  # For base

# Clear plots
dev.off()  # But only if there IS a plot

# Clear console
cat("\014")  # ctrl+L

# Clear mind :)

```



# Sec2: hands-on dplyr tut

#### Dplyr

# Introduction to dplyr for Faster Data Manipulation in R

Note: There is a 40-minute [video tutorial](https://www.youtube.com/watch?v=jWjqLW-u3hc) on YouTube that walks through this document in detail.

## Why do I use dplyr?

- Great for data exploration and transformation
- Intuitive to write and easy to read, especially when using the “chaining” syntax (covered below)
- Fast on data frames

## dplyr functionality

- Five basic verbs: `filter`, `select`, `arrange`, `mutate`, `summarise` (plus `group_by`)
- Can work with data stored in databases and [data tables](http://datatable.r-forge.r-project.org/)
- Joins: inner join, left join, semi-join, anti-join (not covered below)
- Window functions for calculating ranking, offsets, and more
- [Better than plyr](http://blog.rstudio.org/2014/01/17/introducing-dplyr/) if you’re only working with data frames (though it doesn’t yet duplicate all of the plyr functionality)
- Examples below are based upon the [latest release](https://github.com/hadley/dplyr/releases), version 0.2 (released May 2014)

## Loading dplyr and an example dataset

- dplyr will mask a few base functions
- If you also use plyr, load plyr first
- hflights is flights departing from two Houston airports in 2011

```R
# load packages
suppressMessages(library(dplyr))
library(hflights)

# explore data
data(hflights)
head(hflights)
##      Year Month DayofMonth DayOfWeek DepTime ArrTime UniqueCarrier
## 5424 2011     1          1         6    1400    1500            AA
## 5425 2011     1          2         7    1401    1501            AA
## 5426 2011     1          3         1    1352    1502            AA
## 5427 2011     1          4         2    1403    1513            AA
## 5428 2011     1          5         3    1405    1507            AA
## 5429 2011     1          6         4    1359    1503            AA
##      FlightNum TailNum ActualElapsedTime AirTime ArrDelay DepDelay Origin
## 5424       428  N576AA                60      40      -10        0    IAH
## 5425       428  N557AA                60      45       -9        1    IAH
## 5426       428  N541AA                70      48       -8       -8    IAH
## 5427       428  N403AA                70      39        3        3    IAH
## 5428       428  N492AA                62      44       -3        5    IAH
## 5429       428  N262AA                64      45       -7       -1    IAH
##      Dest Distance TaxiIn TaxiOut Cancelled CancellationCode Diverted
## 5424  DFW      224      7      13         0                         0
## 5425  DFW      224      6       9         0                         0
## 5426  DFW      224      5      17         0                         0
## 5427  DFW      224      9      22         0                         0
## 5428  DFW      224      9       9         0                         0
## 5429  DFW      224      6      13         0                         0
```

- `tbl_df` creates a “local data frame”
- Local data frame is simply a wrapper for a data frame that prints nicely

```R
# convert to local data frame
flights <- tbl_df(hflights)

# printing only shows 10 rows and as many columns as can fit on your screen
flights
## Source: local data frame [227,496 x 21]
## 
##      Year Month DayofMonth DayOfWeek DepTime ArrTime UniqueCarrier
## 5424 2011     1          1         6    1400    1500            AA
## 5425 2011     1          2         7    1401    1501            AA
## 5426 2011     1          3         1    1352    1502            AA
## 5427 2011     1          4         2    1403    1513            AA
## 5428 2011     1          5         3    1405    1507            AA
## 5429 2011     1          6         4    1359    1503            AA
## 5430 2011     1          7         5    1359    1509            AA
## 5431 2011     1          8         6    1355    1454            AA
## 5432 2011     1          9         7    1443    1554            AA
## 5433 2011     1         10         1    1443    1553            AA
## ..    ...   ...        ...       ...     ...     ...           ...
## Variables not shown: FlightNum (int), TailNum (chr), ActualElapsedTime
##   (int), AirTime (int), ArrDelay (int), DepDelay (int), Origin (chr), Dest
##   (chr), Distance (int), TaxiIn (int), TaxiOut (int), Cancelled (int),
##   CancellationCode (chr), Diverted (int)
# you can specify that you want to see more rows
print(flights, n=20)

# convert to a normal data frame to see all of the columns
data.frame(head(flights))
```

## filter: Keep rows matching criteria

- Base R approach to filtering forces you to repeat the data frame’s name
- dplyr approach is simpler to write and read
- Command structure (for all dplyr verbs):
  - first argument is a data frame
  - return value is a data frame
  - nothing is modified in place
- Note: dplyr generally does not preserve row names

```R
# base R approach to view all flights on January 1
flights[flights$Month==1 & flights$DayofMonth==1, ]
# dplyr approach
# note: you can use comma or ampersand to represent AND condition
filter(flights, Month==1, DayofMonth==1)
## Source: local data frame [552 x 21]
## 
##    Year Month DayofMonth DayOfWeek DepTime ArrTime UniqueCarrier FlightNum
## 1  2011     1          1         6    1400    1500            AA       428
## 2  2011     1          1         6     728     840            AA       460
## 3  2011     1          1         6    1631    1736            AA      1121
## 4  2011     1          1         6    1756    2112            AA      1294
## 5  2011     1          1         6    1012    1347            AA      1700
## 6  2011     1          1         6    1211    1325            AA      1820
## 7  2011     1          1         6     557     906            AA      1994
## 8  2011     1          1         6    1824    2106            AS       731
## 9  2011     1          1         6     654    1124            B6       620
## 10 2011     1          1         6    1639    2110            B6       622
## ..  ...   ...        ...       ...     ...     ...           ...       ...
## Variables not shown: TailNum (chr), ActualElapsedTime (int), AirTime
##   (int), ArrDelay (int), DepDelay (int), Origin (chr), Dest (chr),
##   Distance (int), TaxiIn (int), TaxiOut (int), Cancelled (int),
##   CancellationCode (chr), Diverted (int)
# use pipe for OR condition
filter(flights, UniqueCarrier=="AA" | UniqueCarrier=="UA")
## Source: local data frame [5,316 x 21]
## 
##    Year Month DayofMonth DayOfWeek DepTime ArrTime UniqueCarrier FlightNum
## 1  2011     1          1         6    1400    1500            AA       428
## 2  2011     1          2         7    1401    1501            AA       428
## 3  2011     1          3         1    1352    1502            AA       428
## 4  2011     1          4         2    1403    1513            AA       428
## 5  2011     1          5         3    1405    1507            AA       428
## 6  2011     1          6         4    1359    1503            AA       428
## 7  2011     1          7         5    1359    1509            AA       428
## 8  2011     1          8         6    1355    1454            AA       428
## 9  2011     1          9         7    1443    1554            AA       428
## 10 2011     1         10         1    1443    1553            AA       428
## ..  ...   ...        ...       ...     ...     ...           ...       ...
## Variables not shown: TailNum (chr), ActualElapsedTime (int), AirTime
##   (int), ArrDelay (int), DepDelay (int), Origin (chr), Dest (chr),
##   Distance (int), TaxiIn (int), TaxiOut (int), Cancelled (int),
##   CancellationCode (chr), Diverted (int)
# you can also use %in% operator
filter(flights, UniqueCarrier %in% c("AA", "UA"))
```

## select: Pick columns by name

- Base R approach is awkward to type and to read
- dplyr approach uses similar syntax to `filter`
- Like a SELECT in SQL

```R
# base R approach to select DepTime, ArrTime, and FlightNum columns
flights[, c("DepTime", "ArrTime", "FlightNum")]
# dplyr approach
select(flights, DepTime, ArrTime, FlightNum)
## Source: local data frame [227,496 x 3]
## 
##      DepTime ArrTime FlightNum
## 5424    1400    1500       428
## 5425    1401    1501       428
## 5426    1352    1502       428
## 5427    1403    1513       428
## 5428    1405    1507       428
## 5429    1359    1503       428
## 5430    1359    1509       428
## 5431    1355    1454       428
## 5432    1443    1554       428
## 5433    1443    1553       428
## ..       ...     ...       ...
# use colon to select multiple contiguous columns, and use `contains` to match columns by name
# note: `starts_with`, `ends_with`, and `matches` (for regular expressions) can also be used to match columns by name
select(flights, Year:DayofMonth, contains("Taxi"), contains("Delay"))
 keep all columbns from year to dayofmonth, and all the column cotains some words
## Source: local data frame [227,496 x 7]
## 
##      Year Month DayofMonth TaxiIn TaxiOut ArrDelay DepDelay
## 5424 2011     1          1      7      13      -10        0
## 5425 2011     1          2      6       9       -9        1
## 5426 2011     1          3      5      17       -8       -8
## 5427 2011     1          4      9      22        3        3
## 5428 2011     1          5      9       9       -3        5
## 5429 2011     1          6      6      13       -7       -1
## 5430 2011     1          7     12      15       -1       -1
## 5431 2011     1          8      7      12      -16       -5
## 5432 2011     1          9      8      22       44       43
## 5433 2011     1         10      6      19       43       43
## ..    ...   ...        ...    ...     ...      ...      ...
```

## “Chaining” or “Pipelining”

- Usual way to perform multiple operations in one line is by nesting
- Can write commands in a natural order by using the `%>%` infix operator (which can be pronounced as “then”)

```R
# nesting method to select UniqueCarrier and DepDelay columns and filter for delays over 60 minutes
filter(select(flights, UniqueCarrier, DepDelay), DepDelay > 60)
# chaining method BETTER %>% = THEN 
flights %>%
    select(UniqueCarrier, DepDelay) %>%
    filter(DepDelay > 60)
## Source: local data frame [10,242 x 2]
## 
##    UniqueCarrier DepDelay
## 1             AA       90
## 2             AA       67
## 3             AA       74
## 4             AA      125
## 5             AA       82
## 6             AA       99
## 7             AA       70
## 8             AA       61
## 9             AA       74
## 10            AS       73
## ..           ...      ...
```

- Chaining increases readability significantly when there are many commands
- Operator is automatically imported from the [magrittr](https://github.com/smbache/magrittr) package
- Can be used to replace nesting in R commands outside of dplyr

```R
# create two vectors and calculate Euclidian distance between them
x1 <- 1:5; x2 <- 2:6
sqrt(sum((x1-x2)^2))
# chaining method
(x1-x2)^2 %>% sum() %>% sqrt()
## [1] 2.236
```

## arrange: Reorder rows

```R
# base R approach to select UniqueCarrier and DepDelay columns and sort by DepDelay
flights[order(flights$DepDelay), c("UniqueCarrier", "DepDelay")]
# dplyr approach
flights %>%
    select(UniqueCarrier, DepDelay) %>%
    arrange(DepDelay)
## Source: local data frame [227,496 x 2]
## 
##    UniqueCarrier DepDelay
## 1             OO      -33
## 2             MQ      -23
## 3             XE      -19
## 4             XE      -19
## 5             CO      -18
## 6             EV      -18
## 7             XE      -17
## 8             CO      -17
## 9             XE      -17
## 10            MQ      -17
## ..           ...      ...
# use `desc` for descending
flights %>%
    select(UniqueCarrier, DepDelay) %>%
    arrange(desc(DepDelay))
```

## mutate: Add new variables

- Create new variables that are functions of existing variables

```R
# base R approach to create a new variable Speed (in mph)
flights$Speed <- flights$Distance / flights$AirTime*60
flights[, c("Distance", "AirTime", "Speed")]
# NOT STORED YET
# dplyr approach (prints the new variable but does not store it)
flights %>%
    select(Distance, AirTime) %>%
    mutate(Speed = Distance/AirTime*60)
## Source: local data frame [227,496 x 3]
## 
##    Distance AirTime Speed
## 1       224      40 336.0
## 2       224      45 298.7
## 3       224      48 280.0
## 4       224      39 344.6
## 5       224      44 305.5
## 6       224      45 298.7
## 7       224      43 312.6
## 8       224      40 336.0
## 9       224      41 327.8
## 10      224      45 298.7
## ..      ...     ...   ...
# store the new variable
flights <- flights %>% mutate(Speed = Distance/AirTime*60)
```

## summarise: Reduce variables to values

- Primarily useful with data that has been grouped by one or more variables
- `group_by` creates the groups that will be operated on
- `summarise` uses the provided aggregation function to summarise each group

```R
# base R approaches to calculate the average arrival delay to each destination
head(with(flights, tapply(ArrDelay, Dest, mean, na.rm=TRUE)))
head(aggregate(ArrDelay ~ Dest, flights, mean))
# dplyr approach: create a table grouped by Dest, and then summarise each group by taking the mean of ArrDelay
flights %>%
    group_by(Dest) %>%
    summarise(avg_delay = mean(ArrDelay, na.rm=TRUE))
## Source: local data frame [116 x 2]
## 
##    Dest avg_delay
## 1   ABQ     7.226
## 2   AEX     5.839
## 3   AGS     4.000
## 4   AMA     6.840
## 5   ANC    26.081
## 6   ASE     6.795
## 7   ATL     8.233
## 8   AUS     7.449
## 9   AVL     9.974
## 10  BFL   -13.199
## ..  ...       ...
```

- `summarise_each` allows you to apply the same summary function to multiple columns at once
- Note: `mutate_each` is also available

```R
# for each carrier, calculate the percentage of flights cancelled or diverted
# FOR EACH SOMETHING, USUALLY WIRTTEN IN GROUP BY
flights %>%
    group_by(UniqueCarrier) %>%
    summarise_each(funs(mean), Cancelled, Diverted)
# DEFINE THE FUNCTION U WANT TO RUN, THEN SELECT THE DATA 
## Source: local data frame [15 x 3]
## 
##    UniqueCarrier Cancelled Diverted
## 1             AA  0.018496 0.001850
## 2             AS  0.000000 0.002740
## 3             B6  0.025899 0.005755
## 4             CO  0.006783 0.002627
## 5             DL  0.015903 0.003029
## 6             EV  0.034483 0.003176
## 7             F9  0.007160 0.000000
## 8             FL  0.009818 0.003273
## 9             MQ  0.029045 0.001936
## 10            OO  0.013947 0.003487
## 11            UA  0.016409 0.002413
## 12            US  0.011269 0.001470
## 13            WN  0.015504 0.002294
## 14            XE  0.015496 0.003450
## 15            YV  0.012658 0.000000
# for each carrier, calculate the minimum and maximum arrival and departure delays
flights %>%
    group_by(UniqueCarrier) %>%
    summarise_each(funs(min(., na.rm=TRUE), max(., na.rm=TRUE)), matches("Delay"))
## DOT IS THE PLACE HOLDER, AND THE NA.RM STANDS FOR REMOVE ALL NA DATA, 
## Source: local data frame [15 x 5]
## 
##    UniqueCarrier ArrDelay_min DepDelay_min ArrDelay_max DepDelay_max
## 1             AA          -39          -15          978          970
## 2             AS          -43          -15          183          172
## 3             B6          -44          -14          335          310
## 4             CO          -55          -18          957          981
## 5             DL          -32          -17          701          730
## 6             EV          -40          -18          469          479
## 7             F9          -24          -15          277          275
## 8             FL          -30          -14          500          507
## 9             MQ          -38          -23          918          931
## 10            OO          -57          -33          380          360
## 11            UA          -47          -11          861          869
## 12            US          -42          -17          433          425
## 13            WN          -44          -10          499          548
## 14            XE          -70          -19          634          628
## 15            YV          -32          -11           72           54
```

- Helper function `n()` counts the number of rows in a group
- Helper function `n_distinct(vector)` counts the number of unique items in that vector

```R
# for each day of the year, count the total number of flights and sort in descending order
flights %>%
    group_by(Month, DayofMonth) %>%
    summarise(flight_count = n()) %>%
    arrange(desc(flight_count))
## Source: local data frame [365 x 3]
## Groups: Month
## 
##    Month DayofMonth flight_count
## 1      8          4          706
## 2      8         11          706
## 3      8         12          706
## 4      8          5          705
## 5      8          3          704
## 6      8         10          704
## 7      1          3          702
## 8      7          7          702
## 9      7         14          702
## 10     7         28          701
## ..   ...        ...          ...
# rewrite more simply with the `tally` function
flights %>%
    group_by(Month, DayofMonth) %>%
    tally(sort = TRUE)
## Source: local data frame [365 x 3]
## Groups: Month
## 
##    Month DayofMonth   n
## 1      8          4 706
## 2      8         11 706
## 3      8         12 706
## 4      8          5 705
## 5      8          3 704
## 6      8         10 704
## 7      1          3 702
## 8      7          7 702
## 9      7         14 702
## 10     7         28 701
## ..   ...        ... ...
# for each destination, count the total number of flights and the number of distinct planes that flew there
flights %>%
    group_by(Dest) %>%
    summarise(flight_count = n(), plane_count = n_distinct(TailNum))
## Source: local data frame [116 x 3]
## 
##    Dest flight_count plane_count
## 1   ABQ         2812         716
## 2   AEX          724         215
## 3   AGS            1           1
## 4   AMA         1297         158
## 5   ANC          125          38
## 6   ASE          125          60
## 7   ATL         7886         983
## 8   AUS         5022        1015
## 9   AVL          350         142
## 10  BFL          504          70
## ..  ...          ...         ...
```

- Grouping can sometimes be useful without summarising

```R
# for each destination, show the number of cancelled and not cancelled flights
flights %>%
    group_by(Dest) %>%
    select(Cancelled) %>%
    table() %>%
    head()
##      Cancelled
## Dest     0  1
##   ABQ 2787 25
##   AEX  712 12
##   AGS    1  0
##   AMA 1265 32
##   ANC  125  0
##   ASE  120  5
```

## Window Functions

- Aggregation function (like `mean`) takes n inputs and returns 1 value

- Window function

   

  takes n inputs and returns n values

  - Includes ranking and ordering functions (like `min_rank`), offset functions (`lead` and `lag`), and cumulative aggregates (like `cummean`).

```R
# for each carrier, calculate which two days of the year they had their longest departure delays
# note: smallest (not largest) value is ranked as 1, so you have to use `desc` to rank by largest value
flights %>%
    group_by(UniqueCarrier) %>%
    select(Month, DayofMonth, DepDelay) %>%
    filter(min_rank(desc(DepDelay)) <= 2) %>%
    arrange(UniqueCarrier, desc(DepDelay))
# rewrite more simply with the `top_n` function
flights %>%
    group_by(UniqueCarrier) %>%
    select(Month, DayofMonth, DepDelay) %>%
    top_n(2) %>%
    arrange(UniqueCarrier, desc(DepDelay))
## Selecting by DepDelay
## Source: local data frame [30 x 4]
## Groups: UniqueCarrier
## 
##    UniqueCarrier Month DayofMonth DepDelay
## 1             AA    12         12      970
## 2             AA    11         19      677
## 3             AS     2         28      172
## 4             AS     7          6      138
## 5             B6    10         29      310
## 6             B6     8         19      283
## 7             CO     8          1      981
## 8             CO     1         20      780
## 9             DL    10         25      730
## 10            DL     4          5      497
## 11            EV     6         25      479
## 12            EV     1          5      465
## 13            F9     5         12      275
## 14            F9     5         20      240
## 15            FL     2         19      507
## 16            FL     3         14      493
## 17            MQ    11          8      931
## 18            MQ     6          9      814
## 19            OO     2         27      360
## 20            OO     4          4      343
## 21            UA     6         21      869
## 22            UA     9         18      588
## 23            US     4         19      425
## 24            US     8         26      277
## 25            WN     4          8      548
## 26            WN     9         29      503
## 27            XE    12         29      628
## 28            XE    12         29      511
## 29            YV     4         22       54
## 30            YV     4         30       46
# for each month, calculate the number of flights and the change from the previous month
flights %>%
    group_by(Month) %>%
    summarise(flight_count = n()) %>%
    mutate(change = flight_count - lag(flight_count))
## Source: local data frame [12 x 3]
## 
##    Month flight_count change
## 1      1        18910     NA
## 2      2        17128  -1782
## 3      3        19470   2342
## 4      4        18593   -877
## 5      5        19172    579
## 6      6        19600    428
## 7      7        20548    948
## 8      8        20176   -372
## 9      9        18065  -2111
## 10    10        18696    631
## 11    11        18021   -675
## 12    12        19117   1096
# rewrite more simply with the `tally` function
flights %>%
    group_by(Month) %>%
    tally() %>%
    mutate(change = n - lag(n))
## Source: local data frame [12 x 3]
## 
##    Month     n change
## 1      1 18910     NA
## 2      2 17128  -1782
## 3      3 19470   2342
## 4      4 18593   -877
## 5      5 19172    579
## 6      6 19600    428
## 7      7 20548    948
## 8      8 20176   -372
## 9      9 18065  -2111
## 10    10 18696    631
## 11    11 18021   -675
## 12    12 19117   1096
```

## Other Useful Convenience Functions

```R
# randomly sample a fixed number of rows, without replacement
flights %>% sample_n(5)
## Source: local data frame [5 x 22]
## 
##        Year Month DayofMonth DayOfWeek DepTime ArrTime UniqueCarrier
## 182007 2011    10          9         7    1046    1201            XE
## 37861  2011     3         24         4    1035    1348            CO
## 14377  2011     1         21         5     836    1134            XE
## 83665  2011     5         10         2     724    1116            XE
## 36997  2011     3         28         1    1225    1325            CO
## Variables not shown: FlightNum (int), TailNum (chr), ActualElapsedTime
##   (int), AirTime (int), ArrDelay (int), DepDelay (int), Origin (chr), Dest
##   (chr), Distance (int), TaxiIn (int), TaxiOut (int), Cancelled (int),
##   CancellationCode (chr), Diverted (int), Speed (dbl)
# randomly sample a fraction of rows, with replacement
flights %>% sample_frac(0.25, replace=TRUE)
## Source: local data frame [56,874 x 22]
## 
##        Year Month DayofMonth DayOfWeek DepTime ArrTime UniqueCarrier
## 38972  2011     3         18         5    1331    1719            CO
## 123045 2011     7         25         1    1729    2113            XE
## 116778 2011     7         14         4    1757    2217            CO
## 215788 2011    12          8         4    2100    2152            WN
## 142781 2011     8          8         1    1411    1749            XE
## 147576 2011     8         31         3    1848    2157            CO
## 69130  2011     4         10         7    1141    1216            XE
## 64357  2011     4         29         5     856    1018            WN
## 184875 2011    10         24         1    1524    1711            XE
## 124623 2011     7          5         2    1406    1720            XE
## ..      ...   ...        ...       ...     ...     ...           ...
## Variables not shown: FlightNum (int), TailNum (chr), ActualElapsedTime
##   (int), AirTime (int), ArrDelay (int), DepDelay (int), Origin (chr), Dest
##   (chr), Distance (int), TaxiIn (int), TaxiOut (int), Cancelled (int),
##   CancellationCode (chr), Diverted (int), Speed (dbl)
# base R approach to view the structure of an object
str(flights)
## Classes 'tbl_df', 'tbl' and 'data.frame':    227496 obs. of  22 variables:
##  $ Year             : int  2011 2011 2011 2011 2011 2011 2011 2011 2011 2011 ...
##  $ Month            : int  1 1 1 1 1 1 1 1 1 1 ...
##  $ DayofMonth       : int  1 2 3 4 5 6 7 8 9 10 ...
##  $ DayOfWeek        : int  6 7 1 2 3 4 5 6 7 1 ...
##  $ DepTime          : int  1400 1401 1352 1403 1405 1359 1359 1355 1443 1443 ...
##  $ ArrTime          : int  1500 1501 1502 1513 1507 1503 1509 1454 1554 1553 ...
##  $ UniqueCarrier    : chr  "AA" "AA" "AA" "AA" ...
##  $ FlightNum        : int  428 428 428 428 428 428 428 428 428 428 ...
##  $ TailNum          : chr  "N576AA" "N557AA" "N541AA" "N403AA" ...
##  $ ActualElapsedTime: int  60 60 70 70 62 64 70 59 71 70 ...
##  $ AirTime          : int  40 45 48 39 44 45 43 40 41 45 ...
##  $ ArrDelay         : int  -10 -9 -8 3 -3 -7 -1 -16 44 43 ...
##  $ DepDelay         : int  0 1 -8 3 5 -1 -1 -5 43 43 ...
##  $ Origin           : chr  "IAH" "IAH" "IAH" "IAH" ...
##  $ Dest             : chr  "DFW" "DFW" "DFW" "DFW" ...
##  $ Distance         : int  224 224 224 224 224 224 224 224 224 224 ...
##  $ TaxiIn           : int  7 6 5 9 9 6 12 7 8 6 ...
##  $ TaxiOut          : int  13 9 17 22 9 13 15 12 22 19 ...
##  $ Cancelled        : int  0 0 0 0 0 0 0 0 0 0 ...
##  $ CancellationCode : chr  "" "" "" "" ...
##  $ Diverted         : int  0 0 0 0 0 0 0 0 0 0 ...
##  $ Speed            : num  336 299 280 345 305 ...
# dplyr approach: better formatting, and adapts to your screen width
glimpse(flights)
## Variables:
## $ Year              (int) 2011, 2011, 2011, 2011, 2011, 2011, 2011, 20...
## $ Month             (int) 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,...
## $ DayofMonth        (int) 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 1...
## $ DayOfWeek         (int) 6, 7, 1, 2, 3, 4, 5, 6, 7, 1, 2, 3, 4, 5, 6,...
## $ DepTime           (int) 1400, 1401, 1352, 1403, 1405, 1359, 1359, 13...
## $ ArrTime           (int) 1500, 1501, 1502, 1513, 1507, 1503, 1509, 14...
## $ UniqueCarrier     (chr) "AA", "AA", "AA", "AA", "AA", "AA", "AA", "A...
## $ FlightNum         (int) 428, 428, 428, 428, 428, 428, 428, 428, 428,...
## $ TailNum           (chr) "N576AA", "N557AA", "N541AA", "N403AA", "N49...
## $ ActualElapsedTime (int) 60, 60, 70, 70, 62, 64, 70, 59, 71, 70, 70, ...
## $ AirTime           (int) 40, 45, 48, 39, 44, 45, 43, 40, 41, 45, 42, ...
## $ ArrDelay          (int) -10, -9, -8, 3, -3, -7, -1, -16, 44, 43, 29,...
## $ DepDelay          (int) 0, 1, -8, 3, 5, -1, -1, -5, 43, 43, 29, 19, ...
## $ Origin            (chr) "IAH", "IAH", "IAH", "IAH", "IAH", "IAH", "I...
## $ Dest              (chr) "DFW", "DFW", "DFW", "DFW", "DFW", "DFW", "D...
## $ Distance          (int) 224, 224, 224, 224, 224, 224, 224, 224, 224,...
## $ TaxiIn            (int) 7, 6, 5, 9, 9, 6, 12, 7, 8, 6, 8, 4, 6, 5, 6...
## $ TaxiOut           (int) 13, 9, 17, 22, 9, 13, 15, 12, 22, 19, 20, 11...
## $ Cancelled         (int) 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,...
## $ CancellationCode  (chr) "", "", "", "", "", "", "", "", "", "", "", ...
## $ Diverted          (int) 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,...
## $ Speed             (dbl) 336.0, 298.7, 280.0, 344.6, 305.5, 298.7, 31...
```

## Connecting to Databases

- dplyr can connect to a database as if the data was loaded into a data frame
- Use the same syntax for local data frames and databases
- Only generates SELECT statements
- Currently supports SQLite, PostgreSQL/Redshift, MySQL/MariaDB, BigQuery, MonetDB
- Example below is based upon an SQLite database containing the hflights data
  - Instructions for creating this database are in the [databases vignette](http://cran.r-project.org/web/packages/dplyr/vignettes/databases.html)

```R
# connect to an SQLite database containing the hflights data
my_db <- src_sqlite("my_db.sqlite3")
## Loading required package: RSQLite
## Loading required package: DBI
## Loading required package: RSQLite.extfuns
# connect to the "hflights" table in that database
flights_tbl <- tbl(my_db, "hflights")

# example query with our data frame
flights %>%
    select(UniqueCarrier, DepDelay) %>%
    arrange(desc(DepDelay))
## Source: local data frame [227,496 x 2]
## 
##    UniqueCarrier DepDelay
## 1             CO      981
## 2             AA      970
## 3             MQ      931
## 4             UA      869
## 5             MQ      814
## 6             MQ      803
## 7             CO      780
## 8             CO      758
## 9             DL      730
## 10            MQ      691
## ..           ...      ...
# identical query using the database
flights_tbl %>%
    select(UniqueCarrier, DepDelay) %>%
    arrange(desc(DepDelay))
## Source: sqlite 3.7.17 [my_db.sqlite3]
## From: hflights [227,496 x 2]
## Arrange: desc(DepDelay) 
## 
##    UniqueCarrier DepDelay
## 1             CO      981
## 2             AA      970
## 3             MQ      931
## 4             UA      869
## 5             MQ      814
## 6             MQ      803
## 7             CO      780
## 8             CO      758
## 9             DL      730
## 10            MQ      691
## ..           ...      ...
```

- You can write the SQL commands yourself
- dplyr can tell you the SQL it plans to run and the query execution plan

```R
# send SQL commands to the database
tbl(my_db, sql("SELECT * FROM hflights LIMIT 100"))
## Source: sqlite 3.7.17 [my_db.sqlite3]
## From: <derived table> [?? x 21]
## 
##    Year Month DayofMonth DayOfWeek DepTime ArrTime UniqueCarrier FlightNum
## 1  2011     1          1         6    1400    1500            AA       428
## 2  2011     1          2         7    1401    1501            AA       428
## 3  2011     1          3         1    1352    1502            AA       428
## 4  2011     1          4         2    1403    1513            AA       428
## 5  2011     1          5         3    1405    1507            AA       428
## 6  2011     1          6         4    1359    1503            AA       428
## 7  2011     1          7         5    1359    1509            AA       428
## 8  2011     1          8         6    1355    1454            AA       428
## 9  2011     1          9         7    1443    1554            AA       428
## 10 2011     1         10         1    1443    1553            AA       428
## ..  ...   ...        ...       ...     ...     ...           ...       ...
## Variables not shown: TailNum (chr), ActualElapsedTime (int), AirTime
##   (int), ArrDelay (int), DepDelay (int), Origin (chr), Dest (chr),
##   Distance (int), TaxiIn (int), TaxiOut (int), Cancelled (int),
##   CancellationCode (chr), Diverted (int)
# ask dplyr for the SQL commands
flights_tbl %>%
    select(UniqueCarrier, DepDelay) %>%
    arrange(desc(DepDelay)) %>%
    explain()
## <SQL>
## SELECT "UniqueCarrier" AS "UniqueCarrier", "DepDelay" AS "DepDelay"
## FROM "hflights"
## ORDER BY "DepDelay" DESC
## 
## <PLAN>
##   selectid order from                             detail
## 1        0     0    0 SCAN TABLE hflights (~227496 rows)
## 2        0     0    0       USE TEMP B-TREE FOR ORDER BY
```

## Resources

- [Official dplyr reference manual and vignettes on CRAN](http://cran.r-project.org/web/packages/dplyr/index.html): vignettes are well-written and cover many aspects of dplyr
- [July 2014 webinar about dplyr (and ggvis) by Hadley Wickham](http://pages.rstudio.net/Webinar-Series-Recording-Essential-Tools-for-R.html) and related [slides/code](https://github.com/rstudio/webinars/tree/master/2014-01): mostly conceptual, with a bit of code
- [dplyr tutorial by Hadley Wickham](https://www.dropbox.com/sh/i8qnluwmuieicxc/AAAgt9tIKoIm7WZKIyK25lh6a) at the [useR! 2014 conference](http://user2014.stat.ucla.edu/): excellent, in-depth tutorial with lots of example code (Dropbox link includes slides, code files, and data files)
- [dplyr GitHub repo](https://github.com/hadley/dplyr) and [list of releases](https://github.com/hadley/dplyr/releases)

< END OF DOCUMENT >



# Going deeper with dplyr: New features in 0.3 and 0.4

## Introduction

In August 2014, I created a [40-minute video tutorial](https://www.youtube.com/watch?v=jWjqLW-u3hc) introducing the key functionality of the dplyr package in R, using dplyr version 0.2. Since then, there have been two significant updates to dplyr (0.3 and 0.4), introducing a ton of new features.

This document (created in March 2015) covers the most useful new features in 0.3 and 0.4, as well as other functionality that I didn’t cover last time (though it is not necessarily new). My [new video tutorial](https://www.youtube.com/watch?v=2mh1PqfsXVI) walks through the code below in detail.

**If you have not watched the [previous tutorial](https://www.youtube.com/watch?v=jWjqLW-u3hc)**, I recommend you do so first since it covers some dplyr basics that will not be covered in this tutorial.

## Loading dplyr and the nycflights13 dataset

Although my last tutorial used data from the hflights package, Hadley Wickham has rewritten the [dplyr vignettes](http://cran.r-project.org/web/packages/dplyr/index.html) to use the nycflights13 package instead, and so I’m also using nycflights13 for the sake of consistency.

```r
# remove flights data if you just finished my previous tutorial
rm(flights)
# load packages
suppressMessages(library(dplyr))
library(nycflights13)

# print the flights dataset from nycflights13
flights
## Source: local data frame [336,776 x 16]
## 
##    year month day dep_time dep_delay arr_time arr_delay carrier tailnum
## 1  2013     1   1      517         2      830        11      UA  N14228
## 2  2013     1   1      533         4      850        20      UA  N24211
## 3  2013     1   1      542         2      923        33      AA  N619AA
## 4  2013     1   1      544        -1     1004       -18      B6  N804JB
## 5  2013     1   1      554        -6      812       -25      DL  N668DN
## 6  2013     1   1      554        -4      740        12      UA  N39463
## 7  2013     1   1      555        -5      913        19      B6  N516JB
## 8  2013     1   1      557        -3      709       -14      EV  N829AS
## 9  2013     1   1      557        -3      838        -8      B6  N593JB
## 10 2013     1   1      558        -2      753         8      AA  N3ALAA
## ..  ...   ... ...      ...       ...      ...       ...     ...     ...
## Variables not shown: flight (int), origin (chr), dest (chr), air_time
##   (dbl), distance (dbl), hour (dbl), minute (dbl)
```

## Choosing columns: select, rename

```R
# besides just using select() to pick columns...
flights %>% select(carrier, flight)
## Source: local data frame [336,776 x 2]
## 
##    carrier flight
## 1       UA   1545
## 2       UA   1714
## 3       AA   1141
## 4       B6    725
## 5       DL    461
## 6       UA   1696
## 7       B6    507
## 8       EV   5708
## 9       B6     79
## 10      AA    301
## ..     ...    ...
# ...you can use the minus sign to hide columns
flights %>% select(-month, -day)
## Source: local data frame [336,776 x 14]
## 
##    year dep_time dep_delay arr_time arr_delay carrier tailnum flight
## 1  2013      517         2      830        11      UA  N14228   1545
## 2  2013      533         4      850        20      UA  N24211   1714
## 3  2013      542         2      923        33      AA  N619AA   1141
## 4  2013      544        -1     1004       -18      B6  N804JB    725
## 5  2013      554        -6      812       -25      DL  N668DN    461
## 6  2013      554        -4      740        12      UA  N39463   1696
## 7  2013      555        -5      913        19      B6  N516JB    507
## 8  2013      557        -3      709       -14      EV  N829AS   5708
## 9  2013      557        -3      838        -8      B6  N593JB     79
## 10 2013      558        -2      753         8      AA  N3ALAA    301
## ..  ...      ...       ...      ...       ...     ...     ...    ...
## Variables not shown: origin (chr), dest (chr), air_time (dbl), distance
##   (dbl), hour (dbl), minute (dbl)
# hide a range of columns
flights %>% select(-(dep_time:arr_delay))

# hide any column with a matching name
flights %>% select(-contains("time"))
# pick columns using a character vector of column names
cols <- c("carrier", "flight", "tailnum")
flights %>% select(one_of(cols))
## Source: local data frame [336,776 x 3]
## 
##    carrier flight tailnum
## 1       UA   1545  N14228
## 2       UA   1714  N24211
## 3       AA   1141  N619AA
## 4       B6    725  N804JB
## 5       DL    461  N668DN
## 6       UA   1696  N39463
## 7       B6    507  N516JB
## 8       EV   5708  N829AS
## 9       B6     79  N593JB
## 10      AA    301  N3ALAA
## ..     ...    ...     ...
# select() can be used to rename columns, though all columns not mentioned are dropped
flights %>% select(tail = tailnum)
## Source: local data frame [336,776 x 1]
## 
##      tail
## 1  N14228
## 2  N24211
## 3  N619AA
## 4  N804JB
## 5  N668DN
## 6  N39463
## 7  N516JB
## 8  N829AS
## 9  N593JB
## 10 N3ALAA
## ..    ...
# rename() does the same thing, except all columns not mentioned are kept
flights %>% rename(tail = tailnum)
## Source: local data frame [336,776 x 16]
## 
##    year month day dep_time dep_delay arr_time arr_delay carrier   tail
## 1  2013     1   1      517         2      830        11      UA N14228
## 2  2013     1   1      533         4      850        20      UA N24211
## 3  2013     1   1      542         2      923        33      AA N619AA
## 4  2013     1   1      544        -1     1004       -18      B6 N804JB
## 5  2013     1   1      554        -6      812       -25      DL N668DN
## 6  2013     1   1      554        -4      740        12      UA N39463
## 7  2013     1   1      555        -5      913        19      B6 N516JB
## 8  2013     1   1      557        -3      709       -14      EV N829AS
## 9  2013     1   1      557        -3      838        -8      B6 N593JB
## 10 2013     1   1      558        -2      753         8      AA N3ALAA
## ..  ...   ... ...      ...       ...      ...       ...     ...    ...
## Variables not shown: flight (int), origin (chr), dest (chr), air_time
##   (dbl), distance (dbl), hour (dbl), minute (dbl)
```

## Choosing rows: filter, between, slice, sample_n, top_n, distinct

```R
# filter() supports the use of multiple conditions
flights %>% filter(dep_time >= 600, dep_time <= 605)
## Source: local data frame [2,460 x 16]
## 
##    year month day dep_time dep_delay arr_time arr_delay carrier tailnum
## 1  2013     1   1      600         0      851        -7      B6  N595JB
## 2  2013     1   1      600         0      837        12      MQ  N542MQ
## 3  2013     1   1      601         1      844        -6      B6  N644JB
## 4  2013     1   1      602        -8      812        -8      DL  N971DL
## 5  2013     1   1      602        -3      821        16      MQ  N730MQ
## 6  2013     1   2      600         0      814        25      EV  N13914
## 7  2013     1   2      600        -5      751       -27      EV  N760EV
## 8  2013     1   2      600         0      819         4      9E  N8946A
## 9  2013     1   2      600         0      846         0      B6  N529JB
## 10 2013     1   2      600         0      737        12      WN  N8311Q
## ..  ...   ... ...      ...       ...      ...       ...     ...     ...
## Variables not shown: flight (int), origin (chr), dest (chr), air_time
##   (dbl), distance (dbl), hour (dbl), minute (dbl)
# between() is a concise alternative for determing if numeric values fall in a range
flights %>% filter(between(dep_time, 600, 605))

# side note: is.na() can also be useful when filtering
flights %>% filter(!is.na(dep_time))
# slice() filters rows by position
flights %>% slice(1000:1005)
## Source: local data frame [6 x 16]
## 
##   year month day dep_time dep_delay arr_time arr_delay carrier tailnum
## 1 2013     1   2      809        -1      950         2      B6  N304JB
## 2 2013     1   2      810        10     1008        -6      DL  N358NW
## 3 2013     1   2      811        -4     1100         4      DL  N328NW
## 4 2013     1   2      811        -4     1126        -5      DL  N305DQ
## 5 2013     1   2      811        -9      944       -11      MQ  N509MQ
## 6 2013     1   2      815         0     1109       -19      DL  N335NW
## Variables not shown: flight (int), origin (chr), dest (chr), air_time
##   (dbl), distance (dbl), hour (dbl), minute (dbl)
# keep the first three rows within each group
flights %>% group_by(month, day) %>% slice(1:3)
## Source: local data frame [1,095 x 16]
## Groups: month, day
## 
##    year month day dep_time dep_delay arr_time arr_delay carrier tailnum
## 1  2013     1   1      517         2      830        11      UA  N14228
## 2  2013     1   1      533         4      850        20      UA  N24211
## 3  2013     1   1      542         2      923        33      AA  N619AA
## 4  2013     1   2       42        43      518        36      B6  N580JB
## 5  2013     1   2      126       156      233       154      B6  N636JB
## 6  2013     1   2      458        -2      703        13      US  N162UW
## 7  2013     1   3       32        33      504        22      B6  N763JB
## 8  2013     1   3       50       185      203       172      B6  N329JB
## 9  2013     1   3      235       156      700       143      B6  N618JB
## 10 2013     1   4       25        26      505        23      B6  N554JB
## ..  ...   ... ...      ...       ...      ...       ...     ...     ...
## Variables not shown: flight (int), origin (chr), dest (chr), air_time
##   (dbl), distance (dbl), hour (dbl), minute (dbl)
# sample three rows from each group
flights %>% group_by(month, day) %>% sample_n(3)
## Source: local data frame [1,095 x 16]
## Groups: month, day
## 
##    year month day dep_time dep_delay arr_time arr_delay carrier tailnum
## 1  2013     1   1     1521         6     1830         7      DL  N378NW
## 2  2013     1   1     1854        24     2055        40      MQ  N518MQ
## 3  2013     1   1     1915        -5     2238       -19      DL  N633DL
## 4  2013     1   2     1550         0     1926         1      DL  N633DL
## 5  2013     1   2     1806         5     2140         3      UA  N12116
## 6  2013     1   2      925         5     1124        16      B6  N239JB
## 7  2013     1   3     2154        -1       50         9      B6  N508JB
## 8  2013     1   3     1448        -2     1631        -9      MQ  N835MQ
## 9  2013     1   3     1651         1     1843       -53      UA  N423UA
## 10 2013     1   4     2203        88     2309        87      EV  N22909
## ..  ...   ... ...      ...       ...      ...       ...     ...     ...
## Variables not shown: flight (int), origin (chr), dest (chr), air_time
##   (dbl), distance (dbl), hour (dbl), minute (dbl)
# keep three rows from each group with the top dep_delay
flights %>% group_by(month, day) %>% (3, dep_delay)
## Source: local data frame [1,108 x 16]
## Groups: month, day
## 
##    year month day dep_time dep_delay arr_time arr_delay carrier tailnum
## 1  2013     1   1      848       853     1001       851      MQ  N942MQ
## 2  2013     1   1     1815       290     2120       338      EV  N17185
## 3  2013     1   1     2343       379      314       456      EV  N21197
## 4  2013     1   2     1412       334     1710       323      UA  N474UA
## 5  2013     1   2     1607       337     2003       368      AA  N324AA
## 6  2013     1   2     2131       379     2340       359      UA  N593UA
## 7  2013     1   3     2008       268     2339       270      DL  N338NW
## 8  2013     1   3     2012       252     2314       257      B6  N558JB
## 9  2013     1   3     2056       291     2239       285      9E  N928XJ
## 10 2013     1   4     2058       208        2       172      B6  N523JB
## ..  ...   ... ...      ...       ...      ...       ...     ...     ...
## Variables not shown: flight (int), origin (chr), dest (chr), air_time
##   (dbl), distance (dbl), hour (dbl), minute (dbl)
# also sort by dep_delay within each group
flights %>% group_by(month, day) %>% top_n(3, dep_delay) %>% arrange(desc(dep_delay))
## Source: local data frame [1,108 x 16]
## Groups: month, day
## 
##    year month day dep_time dep_delay arr_time arr_delay carrier tailnum
## 1  2013     1   1      848       853     1001       851      MQ  N942MQ
## 2  2013     1   1     2343       379      314       456      EV  N21197
## 3  2013     1   1     1815       290     2120       338      EV  N17185
## 4  2013     1   2     2131       379     2340       359      UA  N593UA
## 5  2013     1   2     1607       337     2003       368      AA  N324AA
## 6  2013     1   2     1412       334     1710       323      UA  N474UA
## 7  2013     1   3     2056       291     2239       285      9E  N928XJ
## 8  2013     1   3     2008       268     2339       270      DL  N338NW
## 9  2013     1   3     2012       252     2314       257      B6  N558JB
## 10 2013     1   4     2123       288     2332       276      EV  N29917
## ..  ...   ... ...      ...       ...      ...       ...     ...     ...
## Variables not shown: flight (int), origin (chr), dest (chr), air_time
##   (dbl), distance (dbl), hour (dbl), minute (dbl)
# unique rows can be identified using unique() from base R
flights %>% select(origin, dest) %>% unique()
## Source: local data frame [224 x 2]
## 
##    origin dest
## 1     EWR  IAH
## 2     LGA  IAH
## 3     JFK  MIA
## 4     JFK  BQN
## 5     LGA  ATL
## 6     EWR  ORD
## 7     EWR  FLL
## 8     LGA  IAD
## 9     JFK  MCO
## 10    LGA  ORD
## ..    ...  ...
# dplyr provides an alternative that is more "efficient"
flights %>% select(origin, dest) %>% distinct()

# side note: when chaining, you don't have to include the parentheses if there are no arguments
flights %>% select(origin, dest) %>% distinct
```

## Adding new variables: mutate, transmute, add_rownames

```R
# mutate() creates a new variable (and keeps all existing variables)
flights %>% mutate(speed = distance/air_time*60)
## Source: local data frame [336,776 x 17]
## 
##    year month day dep_time dep_delay arr_time arr_delay carrier tailnum
## 1  2013     1   1      517         2      830        11      UA  N14228
## 2  2013     1   1      533         4      850        20      UA  N24211
## 3  2013     1   1      542         2      923        33      AA  N619AA
## 4  2013     1   1      544        -1     1004       -18      B6  N804JB
## 5  2013     1   1      554        -6      812       -25      DL  N668DN
## 6  2013     1   1      554        -4      740        12      UA  N39463
## 7  2013     1   1      555        -5      913        19      B6  N516JB
## 8  2013     1   1      557        -3      709       -14      EV  N829AS
## 9  2013     1   1      557        -3      838        -8      B6  N593JB
## 10 2013     1   1      558        -2      753         8      AA  N3ALAA
## ..  ...   ... ...      ...       ...      ...       ...     ...     ...
## Variables not shown: flight (int), origin (chr), dest (chr), air_time
##   (dbl), distance (dbl), hour (dbl), minute (dbl), speed (dbl)
# transmute() only keeps the new variables
flights %>% transmute(speed = distance/air_time*60)
## Source: local data frame [336,776 x 1]
## 
##       speed
## 1  370.0441
## 2  374.2731
## 3  408.3750
## 4  516.7213
## 5  394.1379
## 6  287.6000
## 7  404.4304
## 8  259.2453
## 9  404.5714
## 10 318.6957
## ..      ...
# example data frame with row names
mtcars %>% head()
##                    mpg cyl disp  hp drat    wt  qsec vs am gear carb
## Mazda RX4         21.0   6  160 110 3.90 2.620 16.46  0  1    4    4
## Mazda RX4 Wag     21.0   6  160 110 3.90 2.875 17.02  0  1    4    4
## Datsun 710        22.8   4  108  93 3.85 2.320 18.61  1  1    4    1
## Hornet 4 Drive    21.4   6  258 110 3.08 3.215 19.44  1  0    3    1
## Hornet Sportabout 18.7   8  360 175 3.15 3.440 17.02  0  0    3    2
## Valiant           18.1   6  225 105 2.76 3.460 20.22  1  0    3    1
# add_rownames() turns row names into an explicit variable
mtcars %>% add_rownames("model") %>% head()
##               model  mpg cyl disp  hp drat    wt  qsec vs am gear carb
## 1         Mazda RX4 21.0   6  160 110 3.90 2.620 16.46  0  1    4    4
## 2     Mazda RX4 Wag 21.0   6  160 110 3.90 2.875 17.02  0  1    4    4
## 3        Datsun 710 22.8   4  108  93 3.85 2.320 18.61  1  1    4    1
## 4    Hornet 4 Drive 21.4   6  258 110 3.08 3.215 19.44  1  0    3    1
## 5 Hornet Sportabout 18.7   8  360 175 3.15 3.440 17.02  0  0    3    2
## 6           Valiant 18.1   6  225 105 2.76 3.460 20.22  1  0    3    1
# side note: dplyr no longer prints row names (ever) for local data frames
mtcars %>% tbl_df()
## Source: local data frame [32 x 11]
## 
##     mpg cyl  disp  hp drat    wt  qsec vs am gear carb
## 1  21.0   6 160.0 110 3.90 2.620 16.46  0  1    4    4
## 2  21.0   6 160.0 110 3.90 2.875 17.02  0  1    4    4
## 3  22.8   4 108.0  93 3.85 2.320 18.61  1  1    4    1
## 4  21.4   6 258.0 110 3.08 3.215 19.44  1  0    3    1
## 5  18.7   8 360.0 175 3.15 3.440 17.02  0  0    3    2
## 6  18.1   6 225.0 105 2.76 3.460 20.22  1  0    3    1
## 7  14.3   8 360.0 245 3.21 3.570 15.84  0  0    3    4
## 8  24.4   4 146.7  62 3.69 3.190 20.00  1  0    4    2
## 9  22.8   4 140.8  95 3.92 3.150 22.90  1  0    4    2
## 10 19.2   6 167.6 123 3.92 3.440 18.30  1  0    4    4
## ..  ... ...   ... ...  ...   ...   ... .. ..  ...  ...
```

## Grouping and counting: summarise, tally, count, group_size, n_groups, ungroup

```R
# summarise() can be used to count the number of rows in each group
flights %>% group_by(month) %>% summarise(cnt = n())
## Source: local data frame [12 x 2]
## 
##    month   cnt
## 1      1 27004
## 2      2 24951
## 3      3 28834
## 4      4 28330
## 5      5 28796
## 6      6 28243
## 7      7 29425
## 8      8 29327
## 9      9 27574
## 10    10 28889
## 11    11 27268
## 12    12 28135
# tally() and count() can do this more concisely
flights %>% group_by(month) %>% tally()
flights %>% count(month)
# you can sort by the count
flights %>% group_by(month) %>% summarise(cnt = n()) %>% arrange(desc(cnt))
## Source: local data frame [12 x 2]
## 
##    month   cnt
## 1      7 29425
## 2      8 29327
## 3     10 28889
## 4      3 28834
## 5      5 28796
## 6      4 28330
## 7      6 28243
## 8     12 28135
## 9      9 27574
## 10    11 27268
## 11     1 27004
## 12     2 24951
# tally() and count() have a sort parameter for this purpose
flights %>% group_by(month) %>% tally(sort=TRUE)
flights %>% count(month, sort=TRUE)
# you can sum over a specific variable instead of simply counting rows
flights %>% group_by(month) %>% summarise(dist = sum(distance))
## Source: local data frame [12 x 2]
## 
##    month     dist
## 1      1 27188805
## 2      2 24975509
## 3      3 29179636
## 4      4 29427294
## 5      5 29974128
## 6      6 29856388
## 7      7 31149199
## 8      8 31149334
## 9      9 28711426
## 10    10 30012086
## 11    11 28639718
## 12    12 29954084
# tally() and count() have a wt parameter for this purpose
flights %>% group_by(month) %>% tally(wt = distance)
flights %>% count(month, wt = distance)
# group_size() returns the counts as a vector
flights %>% group_by(month) %>% group_size()
##  [1] 27004 24951 28834 28330 28796 28243 29425 29327 27574 28889 27268
## [12] 28135
# n_groups() simply reports the number of groups
flights %>% group_by(month) %>% n_groups()
## [1] 12
# group by two variables, summarise, arrange (output is possibly confusing)
flights %>% group_by(month, day) %>% summarise(cnt = n()) %>% arrange(desc(cnt)) %>% print(n = 40)
## Source: local data frame [365 x 3]
## Groups: month
## 
##    month day cnt
## 1      1   2 943
## 2      1   7 933
## 3      1  10 932
## 4      1  11 930
## 5      1  14 928
## 6      1  31 928
## 7      1  17 927
## 8      1  24 925
## 9      1  18 924
## 10     1  28 923
## 11     1  25 922
## 12     1   4 915
## 13     1   3 914
## 14     1  21 912
## 15     1   9 902
## 16     1  16 901
## 17     1  30 900
## 18     1   8 899
## 19     1  23 897
## 20     1  15 894
## 21     1  22 890
## 22     1  29 890
## 23     1   1 842
## 24     1   6 832
## 25     1  13 828
## 26     1  27 823
## 27     1  20 786
## 28     1   5 720
## 29     1  12 690
## 30     1  26 680
## 31     1  19 674
## 32     2  28 964
## 33     2  21 961
## 34     2  25 961
## 35     2  22 957
## 36     2  14 956
## 37     2  15 954
## 38     2  20 949
## 39     2  18 948
## 40     2  27 945
## ..   ... ... ...
# ungroup() before arranging to arrange across all groups
flights %>% group_by(month, day) %>% summarise(cnt = n()) %>% ungroup() %>% arrange(desc(cnt))
## Source: local data frame [365 x 3]
## 
##    month day  cnt
## 1     11  27 1014
## 2      7  11 1006
## 3      7   8 1004
## 4      7  10 1004
## 5     12   2 1004
## 6      7  18 1003
## 7      7  25 1003
## 8      7  12 1002
## 9      7   9 1001
## 10     7  17 1001
## ..   ... ...  ...
```

## Creating data frames: data_frame

`data_frame()` is a better way than `data.frame()` for creating data frames. Benefits of `data_frame()`:

- You can use previously defined columns to compute new columns.
- It never coerces column types.
- It never munges column names.
- It never adds row names.
- It only recycles length 1 input.
- It returns a local data frame (a tbl_df).

```r
# data_frame() example
data_frame(a = 1:6, b = a*2, c = 'string', 'd+e' = 1) %>% glimpse()
## Observations: 6
## Variables:
## $ a   (int) 1, 2, 3, 4, 5, 6
## $ b   (dbl) 2, 4, 6, 8, 10, 12
## $ c   (chr) "string", "string", "string", "string", "string", "string"
## $ d+e (dbl) 1, 1, 1, 1, 1, 1
# data.frame() example
data.frame(a = 1:6, c = 'string', 'd+e' = 1) %>% glimpse()
## Observations: 6
## Variables:
## $ a   (int) 1, 2, 3, 4, 5, 6
## $ c   (fctr) string, string, string, string, string, string
## $ d.e (dbl) 1, 1, 1, 1, 1, 1
```

## Joining (merging) tables: left_join, right_join, inner_join, full_join, semi_join, anti_join

```R
# create two simple data frames
(a <- data_frame(color = c("green","yellow","red"), num = 1:3))
## Source: local data frame [3 x 2]
## 
##    color num
## 1  green   1
## 2 yellow   2
## 3    red   3
(b <- data_frame(color = c("green","yellow","pink"), size = c("S","M","L")))
## Source: local data frame [3 x 2]
## 
##    color size
## 1  green    S
## 2 yellow    M
## 3   pink    L
# only include observations found in both "a" and "b" (automatically joins on variables that appear in both tables)
inner_join(a, b)
## Joining by: "color"
## Source: local data frame [2 x 3]
## 
##    color num size
## 1  green   1    S
## 2 yellow   2    M
# include observations found in either "a" or "b"
full_join(a, b)
## Joining by: "color"
## Source: local data frame [4 x 3]
## 
##    color num size
## 1  green   1    S
## 2 yellow   2    M
## 3    red   3   NA
## 4   pink  NA    L
# include all observations found in "a"
left_join(a, b)
## Joining by: "color"
## Source: local data frame [3 x 3]
## 
##    color num size
## 1  green   1    S
## 2 yellow   2    M
## 3    red   3   NA
# include all observations found in "b"
right_join(a, b)
## Joining by: "color"
## Source: local data frame [3 x 3]
## 
##    color num size
## 1  green   1    S
## 2 yellow   2    M
## 3   pink  NA    L
# right_join(a, b) is identical to left_join(b, a) except for column ordering
left_join(b, a)
## Joining by: "color"
## Source: local data frame [3 x 3]
## 
##    color size num
## 1  green    S   1
## 2 yellow    M   2
## 3   pink    L  NA
# filter "a" to only show observations that match "b"
semi_join(a, b)
## Joining by: "color"
## Source: local data frame [2 x 2]
## 
##    color num
## 1  green   1
## 2 yellow   2
# filter "a" to only show observations that don't match "b"
anti_join(a, b)
## Joining by: "color"
## Source: local data frame [1 x 2]
## 
##   color num
## 1   red   3
# sometimes matching variables don't have identical names
b <- b %>% rename(col = color)

# specify that the join should occur by matching "color" in "a" with "col" in "b"
inner_join(a, b, by=c("color" = "col"))
## Source: local data frame [2 x 3]
## 
##    color num size
## 1  green   1    S
## 2 yellow   2    M
```

## Viewing more output: print, View

```R
# specify that you want to see more rows
flights %>% print(n = 15)
## Source: local data frame [336,776 x 16]
## 
##    year month day dep_time dep_delay arr_time arr_delay carrier tailnum
## 1  2013     1   1      517         2      830        11      UA  N14228
## 2  2013     1   1      533         4      850        20      UA  N24211
## 3  2013     1   1      542         2      923        33      AA  N619AA
## 4  2013     1   1      544        -1     1004       -18      B6  N804JB
## 5  2013     1   1      554        -6      812       -25      DL  N668DN
## 6  2013     1   1      554        -4      740        12      UA  N39463
## 7  2013     1   1      555        -5      913        19      B6  N516JB
## 8  2013     1   1      557        -3      709       -14      EV  N829AS
## 9  2013     1   1      557        -3      838        -8      B6  N593JB
## 10 2013     1   1      558        -2      753         8      AA  N3ALAA
## 11 2013     1   1      558        -2      849        -2      B6  N793JB
## 12 2013     1   1      558        -2      853        -3      B6  N657JB
## 13 2013     1   1      558        -2      924         7      UA  N29129
## 14 2013     1   1      558        -2      923       -14      UA  N53441
## 15 2013     1   1      559        -1      941        31      AA  N3DUAA
## ..  ...   ... ...      ...       ...      ...       ...     ...     ...
## Variables not shown: flight (int), origin (chr), dest (chr), air_time
##   (dbl), distance (dbl), hour (dbl), minute (dbl)
# specify that you want to see ALL rows (don't run this!)
flights %>% print(n = Inf)
# specify that you want to see all columns
flights %>% print(width = Inf)
## Source: local data frame [336,776 x 16]
## 
##    year month day dep_time dep_delay arr_time arr_delay carrier tailnum
## 1  2013     1   1      517         2      830        11      UA  N14228
## 2  2013     1   1      533         4      850        20      UA  N24211
## 3  2013     1   1      542         2      923        33      AA  N619AA
## 4  2013     1   1      544        -1     1004       -18      B6  N804JB
## 5  2013     1   1      554        -6      812       -25      DL  N668DN
## 6  2013     1   1      554        -4      740        12      UA  N39463
## 7  2013     1   1      555        -5      913        19      B6  N516JB
## 8  2013     1   1      557        -3      709       -14      EV  N829AS
## 9  2013     1   1      557        -3      838        -8      B6  N593JB
## 10 2013     1   1      558        -2      753         8      AA  N3ALAA
## ..  ...   ... ...      ...       ...      ...       ...     ...     ...
##    flight origin dest air_time distance hour minute
## 1    1545    EWR  IAH      227     1400    5     17
## 2    1714    LGA  IAH      227     1416    5     33
## 3    1141    JFK  MIA      160     1089    5     42
## 4     725    JFK  BQN      183     1576    5     44
## 5     461    LGA  ATL      116      762    5     54
## 6    1696    EWR  ORD      150      719    5     54
## 7     507    EWR  FLL      158     1065    5     55
## 8    5708    LGA  IAD       53      229    5     57
## 9      79    JFK  MCO      140      944    5     57
## 10    301    LGA  ORD      138      733    5     58
## ..    ...    ...  ...      ...      ...  ...    ...
# show up to 1000 rows and all columns
flights %>% View()

# set option to see all columns and fewer rows
options(dplyr.width = Inf, dplyr.print_min = 6)

# reset options (or just close R)
options(dplyr.width = NULL, dplyr.print_min = 10)
```

## Resources

- Release announcements for [version 0.3](http://blog.rstudio.org/2014/10/13/dplyr-0-3-2/) and [version 0.4](http://blog.rstudio.org/2015/01/09/dplyr-0-4-0/)
- [dplyr reference manual and vignettes](http://cran.r-project.org/web/packages/dplyr/)
- [Two-table vignette](http://cran.r-project.org/web/packages/dplyr/vignettes/two-table.html) covering joins and set operations
- [RStudio’s Data Wrangling Cheat Sheet](http://www.rstudio.com/wp-content/uploads/2015/02/data-wrangling-cheatsheet.pdf) for dplyr and tidyr
- [dplyr GitHub repo](https://github.com/hadley/dplyr) and [list of releases](https://github.com/hadley/dplyr/releases)

## Data School

- [Blog](http://www.dataschool.io/)
- [Email newsletter](http://www.dataschool.io/subscribe/)
- [YouTube channel](http://youtube.com/dataschool)

< END OF DOCUMENT >