require "fsp_gen.h";


config var p: c_short = 1;

extern proc generate_flow_shop(x : c_short): void;
extern proc write_problem(x : c_short): void;

writeln(p);


generate_flow_shop(p);                   /* generate problem i  */ 
write_problem(p);  