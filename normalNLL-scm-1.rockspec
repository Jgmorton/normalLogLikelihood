package = "normalNLL"
version = "scm-1"

source = {
   url = "https://github.com/Jgmorton/normalLogLikelihood",
   tag = "master"
}

description = {
   summary = "NLL Loss for Normal/Gaussian mixture distributions",
   detailed = [[
   	    NLL Loss for Normal/Gaussian mixture distributions
   ]],
   homepage = "https://github.com/Jgmorton/normalLogLikelihood"
}

dependencies = {
   "torch >= 7.0"
}

build = {
   type = "command",
   build_command = [[
cmake -E make_directory build;
cd build;
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$(LUA_BINDIR)/.." -DCMAKE_INSTALL_PREFIX="$(PREFIX)"; 
$(MAKE)
   ]],
   install_command = "cd build && $(MAKE) install"
}
