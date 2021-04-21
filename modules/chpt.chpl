module chpt{
	
	use Time;

	record Checkpointer{

		var progress: atomic uint(64);
	  	var partial_tree: atomic uint(64);
	  	var synch_with_checkpointer: atomic bool;
	  	var max_val: int;

	  	proc init(max_val: int) {
	  		this.max_val = max_val;	

  		}//
	
		proc add_to_partial_tree(const val: uint(64)){

			partial_tree.add(val);
		}
	                		
		proc add_to_progress(){
			progress.add(1);
		}

		proc add_to_progress(val: uint(64)){
			progress.add(val);
		}


		proc show_status(current:int, partial_tree: uint(64)){

			var percent = (current:real)/(max_val:real);
			var interval: int  = ((percent*100):int);

			write(" [");

			for x in 0..#interval:int do{
				write("|");
			}

			write("] ");
			write(interval, "% # Nodes remaining: ", max_val-current, " < Partial tree: ", partial_tree," > #\n" );
		}///////////


		proc start(){

			progress.write(0:uint(64));
	  		partial_tree.write(0:uint(64));
	  		synch_with_checkpointer.write(false);    

			var local_value: int;
			local_value = progress.read():int;

			var previous: int = 0;
			var local_tree = partial_tree.read();

			while( local_value==0 ){
				//writeln("Vou dormir");
				sleep(2);
				//writeln("Dormi");
				local_value = progress.read():int;
			}///

			writeln("\n\n## Beginning the checkpointer on locale ", here.id, " ##\n\n");

			while( local_value < max_val ){

				//writeln("Not the max. Going to sleep.");
				if local_value > previous then{
					local_tree = partial_tree.read();
					show_status(local_value,local_tree);
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
		   		sleep(1);
		    	end_of_search = synch_with_checkpointer.read();
			}
		}//wait

	}///register


}//modules