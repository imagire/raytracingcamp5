TARGET=./bin/a.out

CPP=g++
OPT=-std=c++0x -static-libgcc -static-libstdc++ -L. -fopenmp -O3
INC=
LIB=

-include makefile.opt

.SUFFIXES:.cpp .c .o .h

SRC=$(shell ls *.cpp)
HED=$(shell ls *.h)
OBJ=$(SRC:.cpp=.o)

all: $(TARGET)
$(TARGET): $(OBJ)
	ln -s `g++ -print-file-name=libgomp.a`
	$(CPP) $(OPT) -o $(TARGET) $(OBJ) $(LIB)

.c.o:
	$(CPP) $(OPT) -c $< $(INC)
.cpp.o:
	$(CPP) $(OPT) -c $< $(INC)

depend:
	g++ -MM -MG $(SRC) $(OPT) >makefile.depend

clean:
	rm -f $(TARGET) $(TARGET).exe
	rm -f *.o *.obj
	rm -f *~ *.~*

tar:
	tar cvzf $(TARGET).tar.gz $(SRC) $(HED) makefile

-include makefile.depend
