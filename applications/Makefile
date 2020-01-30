APPDIRS = test_app range_detection pulse_doppler

#Prepend x86/arm/clean to each of the appdirs targets so that we can associate different commands with each
APPDIRS_x86 = $(APPDIRS:%=x86-%)
APPDIRS_ARM = $(APPDIRS:%=arm-%)
APPDIRS_CLEAN = $(APPDIRS:%=clean-%)

#Declare them all as phony (not real files on the filesystem)
.PHONY: appdirs x86 arm clean $(APPDIRS) $(APPDIRS_x86) $(APPDIRS_ARM) $(APPDIRS_CLEAN)

#Top level targets so that this makefile accepts options such as `make x86`, `make arm`, and `make clean`
#They depend on their corresponding APPDIRS targets
x86: $(APPDIRS_x86)
arm: $(APPDIRS_ARM)
clean: $(APPDIRS_CLEAN)

$(APPDIRS_x86):
	# Strip the "x86-" prefix off the target name and execute "make x86" in that directory
	$(MAKE) -C $(@:x86-%=%) x86

$(APPDIRS_ARM):
	$(MAKE) -C $(@:arm-%=%) arm

$(APPDIRS_CLEAN):
	$(MAKE) -C $(@:clean-%=%) clean
