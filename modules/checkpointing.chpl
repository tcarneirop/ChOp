module checkpointing{
	
	use Time;


	var progress: atomic uint(64);
	var partial_tree: atomic uint(64);
	var synch_with_checkpointer: atomic bool;

	var start_chkpt: real(64);

	const sleep_time: real = 0.8;

	//@todo partial tree is printing wrong values

	proc start(){
		progress.write(0:uint(64));
	  	partial_tree.write(0:uint(64));
	  	synch_with_checkpointer.write(false);
	  	start_chkpt = getCurrentTime();
	}

	proc show_status(max_val:int, current:int, partial_tree: uint(64)){

		var percent = (current:real)/(max_val:real);
		var interval: int  = ((percent*100):int);

		write(" [");

		for x in 0..#interval:int do{
			write("|");
		}

		write("] ");
		const elapsed = getCurrentTime() - start_chkpt;
		writeln(interval, "% # Nodes remaining: ", max_val-current, "#\n # < Partial tree: ", partial_tree," > < Elapsed time: ",  elapsed," (s) > #");
	}///////////


	proc checkpointer(ref progress: atomic uint(64), ref partial_tree: atomic uint(64), ref synch_with_checkpointer: atomic bool, max_val: int){

		var local_value: int;
		local_value = progress.read():int;

		var previous: int = 0;
		var local_tree = partial_tree.read();

		while( local_value==0 ){
			//writeln("Vou dormir");
			sleep(sleep_time);
			//writeln("Dormi");
			local_value = progress.read():int;
		}///

		writeln("\n\n## Beginning the checkpointer on locale ", here.id, " ##\n\n");

		while( local_value < max_val ){

			//writeln("Not the max. Going to sleep.");
			if local_value > previous then{
				local_tree = partial_tree.read();
				show_status(max_val, local_value,local_tree);
				previous = local_value;
			}
			sleep(1);
			//writeln("Slept.");
			local_value = progress.read():int;
			
		}//

		//show_status(max_val, local_value);
		writeln("\n## Checkpointer is done ##");
		synch_with_checkpointer.write(true);

	}//checkpointer


	proc wait(){

		var end_of_search: bool = synch_with_checkpointer.read();
		while(!end_of_search){
		   	sleep(sleep_time);
		    end_of_search = synch_with_checkpointer.read();
		}//while
		//@endtodo -s if checkpointer

	}//
	

}//modules