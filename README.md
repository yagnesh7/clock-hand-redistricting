# clock-hand-redistricting
## PREVENTING GERRYMANDERING: AN ALGORITHM TO REDISTRICT CONGRESSIONAL DISTRICTS INVULNERABLE TO PARTISAN MANIPULATION

An algorithm proposed by Professor Steven Brams from NYU, this approach attempts to address the gerrymandering of Congressional Districts of the United States by automating the drawing of district lines. Using existing geographic boundaries (e.g., precincts) and voting history, this algorithm draws line by focusing on competitveness. Iterating from the population center of mass, it draws "clockhands" which spans an area that recursively splits the population of the region, starting from the entire state and then ending at individual districts. The algorithm trials many of these population spans while trying to find regions that meet some criteria (e.g., maximizing the competiveness of one side of the span), and then greedily chooses the best performing split and continues this process until the number of desired regions is met.

Please check out the [`redistricting/redistrict_example.ipynb`](https://github.com/yagnesh7/clock-hand-redistricting/blob/main/redistricting/redistrict_example.ipynb) for an example.

Paper currently being written.