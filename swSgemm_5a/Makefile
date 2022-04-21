CC = sw5cc.new
HCFLAGS = -std=gnu99 -O3 -msimd -host
SCFLAGS = -std=gnu99 -O3 -msimd -slave -OPT:alias=disjoint -OPT:IEEE_arithmetic=3 lmfast_slave
LDFLAGS = -lm
HEADER_FILES = myargs.h utils.h

BUILD = ./obj

swSgemm : $(BUILD)/main.o $(BUILD)/utils.o $(BUILD)/master.o $(BUILD)/slave.o
	$(CC) $(HCFLAGS) -hybrid -o $@ $^ $(LDFLAGS) -lm_slave

$(BUILD)/%.o : %.c $(HEADER_FILES)
	$(CC) $(HCFLAGS) -o $@ -c $< 

$(BUILD)/slave.o : slave.c myargs.h
	$(CC) $(SCFLAGS) -o $@ -c $<

run :
	rm -f ./swSgemm.log
	bsub -o ./swSgemm.log -b -q q_sw_share -share_size 6144 -host_stack 1024 -n 1 -cgsp 64 ./swSgemm

clean :
	rm -f $(BUILD)/*.o ./swSgemm
