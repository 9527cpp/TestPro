#.PHONY	= clean
GCC		= cl.exe

INCLUDES	= -I.
CFLAGS		= $(INCLUDES) 
LDFLAGS		= 
CPPOPT = /w /EHsc /D_CRT_SECURE_NO_DEPRECATE
EXE			= main.exe

CPP_SRCS	= main.cpp
C_SRCS = lj_queue_cache.c

CPP_OBJS	= $(CPP_SRCS:.cpp=.obj)
C_OBJS		= $(C_SRCS:.c=.obj)
OBJS        = $(CPP_OBJS) $(C_OBJS)
  
$(EXE):$(OBJS)
	link.exe /out:$(EXE)  $(OBJS)  $(LDFLAGS)
	 
			
$(CPP_OBJS):$(CPP_SRCS)
	$(GCC) $(CPPOPT) /c $(CPP_SRCS)

	
$(C_OBJS):$(C_SRCS)
	$(GCC) $(CPPOPT) /c $(C_SRCS)
		
clean:
	del $(OBJS) $(EXE) 
