# Local Algorithms
PrinPy includes local algorithms for computing principal curves. Local algorithms work by starting at one end of the curve and taking steps towards the end of the curve such that each line it makes meets an error threshold. 

'CLPCG'
***
Implements CLPC-greedy algorithm. 

Methods | Description
------------ | -------------
fit | Calculates principal curve ticks

        Args: x (array): x-data to fit
			  y (array): y-data to fit
			  e_max (flat): Max allowed error. If not met, another point P will 
                  be addedto the curve. Authors suggest 1/4 to 1/2 of 
                  measurement error. Defaults to .2

        Returns: 
            None
Content in the first column | Content in the second column