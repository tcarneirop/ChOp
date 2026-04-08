module queens_call_serial_search{

	use queens_serial;
	use Time; 
    use statistics;
    use CTypes;
    use queens_constants;
    use queens_aux;


	proc queens_call_serial_search(const size: uint(16), const mode: string = "serial", const prepro: bool = false){

		
		var timer: stopwatch;
		var metrics: (uint(64),uint(64));
		timer.start(); // Start timer

		metrics = queens_serial_search(size);

		timer.stop(); // Start timer

		writeln("\nSerial Bitset Queens for size ", size);
		writeln("\tNumber of solutions: ", metrics[0]);
		writeln("\tTree size: ", metrics[1]);
		writeln("Elapsed time: ", timer.elapsed()*1000, " ms"); // Print elapsed time
		writeln("Performance: ", metrics[1]/timer.elapsed(), " n/s\n\n"); // Print elapsed time
		writeln("(performance in subproblems evaluation per sec.): "); // Print elapsed time
        timer.clear(); // Clear timer for parallel loop

	}


}//module
