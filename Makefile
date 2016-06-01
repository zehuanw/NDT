objects := ndt_reduce.o ndt_timer.o 
target := libndt.a
GCC = nvcc
GPU = sm_35


target: $(objects)
	$(GCC) -lib $(objects) -o $(target)

$(objects): %.o: %.cu ndt.h
	$(GCC) -dc -arch=$(GPU) $< -o $@

clean:
	rm $(objects) $(target)

install:
	cp $(target) /usr/lib/ && cp ndt.h /usr/include/
