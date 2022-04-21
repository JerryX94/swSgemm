CC = swgcc
HCFLAGS = -std=gnu99 -O3 -msimd -mhost
SCFLAGS = -std=gnu99 -O3 -msimd -mslave -faddress_align=64 -mfma -fcache-hint
LDFLAGS = -lm
HEADER_FILES = myargs.h utils.h

BUILD = ./obj

swSgemm : $(BUILD)/main.o $(BUILD)/utils.o $(BUILD)/master.o $(BUILD)/slave.o
	$(CC) -mhybrid -o $@ $^ $(LDFLAGS) -lm_slave

$(BUILD)/%.o : %.c $(HEADER_FILES)
	$(CC) $(HCFLAGS) -o $@ -c $< 

$(BUILD)/slave.o : slave.c myargs.h
	$(CC) $(SCFLAGS) -o $@ -c $<

run :
	rm -f ./swSgemm.log
	bsub -J xjy_swSgemm -o ./swSgemm.log -b -q q_sw_expr -share_size 12800 -host_stack 1024 -n 1 -cgsp 64 ./swSgemm

prof_run :
	rm -f ./swSgemm_prof.log
	bsub -J xjy_swSgemm -o ./swSgemm_prof.log -swrunarg "-P master" -b -q q_sw_expr -share_size 12800 -host_stack 1024 -n 1 -cgsp 64 ./swSgemm

clean :
	rm -f $(BUILD)/*.o ./swSgemm
