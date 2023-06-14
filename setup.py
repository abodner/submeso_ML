{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/share/apps/python/3.8.6/intel/lib/python3.8/site-packages/setuptools/distutils_patch.py:25: UserWarning: Distutils was imported before Setuptools. This usage is discouraged and may exhibit undesirable behaviors or errors. Please use Setuptools' objects directly or at least import Setuptools first.\n",
      "  warnings.warn(\n",
      "ERROR:root:Internal Python error in the inspect module.\n",
      "Below is the traceback from this internal error.\n",
      "\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Traceback (most recent call last):\n",
      "  File \"/share/apps/python/3.8.6/intel/lib/python3.8/distutils/fancy_getopt.py\", line 233, in getopt\n",
      "    opts, args = getopt.getopt(args, short_opts, self.long_opts)\n",
      "  File \"/share/apps/python/3.8.6/intel/lib/python3.8/getopt.py\", line 95, in getopt\n",
      "    opts, args = do_shorts(opts, args[0][1:], shortopts, args[1:])\n",
      "  File \"/share/apps/python/3.8.6/intel/lib/python3.8/getopt.py\", line 195, in do_shorts\n",
      "    if short_has_arg(opt, shortopts):\n",
      "  File \"/share/apps/python/3.8.6/intel/lib/python3.8/getopt.py\", line 211, in short_has_arg\n",
      "    raise GetoptError(_('option -%s not recognized') % opt, opt)\n",
      "getopt.GetoptError: option -f not recognized\n",
      "\n",
      "During handling of the above exception, another exception occurred:\n",
      "\n",
      "Traceback (most recent call last):\n",
      "  File \"/share/apps/python/3.8.6/intel/lib/python3.8/distutils/core.py\", line 134, in setup\n",
      "    ok = dist.parse_command_line()\n",
      "  File \"/share/apps/python/3.8.6/intel/lib/python3.8/distutils/dist.py\", line 475, in parse_command_line\n",
      "    args = parser.getopt(args=self.script_args, object=self)\n",
      "  File \"/share/apps/python/3.8.6/intel/lib/python3.8/distutils/fancy_getopt.py\", line 235, in getopt\n",
      "    raise DistutilsArgError(msg)\n",
      "distutils.errors.DistutilsArgError: option -f not recognized\n",
      "\n",
      "During handling of the above exception, another exception occurred:\n",
      "\n",
      "Traceback (most recent call last):\n",
      "  File \"/share/apps/python/3.8.6/intel/lib/python3.8/site-packages/IPython/core/interactiveshell.py\", line 3417, in run_code\n",
      "    exec(code_obj, self.user_global_ns, self.user_ns)\n",
      "  File \"<ipython-input-1-52f8324f9f8e>\", line 8, in <module>\n",
      "    setup(name=\"submeso_ML\",\n",
      "  File \"/share/apps/python/3.8.6/intel/lib/python3.8/site-packages/setuptools/__init__.py\", line 165, in setup\n",
      "    return distutils.core.setup(**attrs)\n",
      "  File \"/share/apps/python/3.8.6/intel/lib/python3.8/distutils/core.py\", line 136, in setup\n",
      "    raise SystemExit(gen_usage(dist.script_name) + \"\\nerror: %s\" % msg)\n",
      "SystemExit: usage: ipykernel_launcher.py [global_opts] cmd1 [cmd1_opts] [cmd2 [cmd2_opts] ...]\n",
      "   or: ipykernel_launcher.py --help [cmd1 cmd2 ...]\n",
      "   or: ipykernel_launcher.py --help-commands\n",
      "   or: ipykernel_launcher.py cmd --help\n",
      "\n",
      "error: option -f not recognized\n",
      "\n",
      "During handling of the above exception, another exception occurred:\n",
      "\n",
      "Traceback (most recent call last):\n",
      "  File \"/share/apps/python/3.8.6/intel/lib/python3.8/site-packages/IPython/core/ultratb.py\", line 1169, in get_records\n",
      "    return _fixed_getinnerframes(etb, number_of_lines_of_context, tb_offset)\n",
      "  File \"/share/apps/python/3.8.6/intel/lib/python3.8/site-packages/IPython/core/ultratb.py\", line 316, in wrapped\n",
      "    return f(*args, **kwargs)\n",
      "  File \"/share/apps/python/3.8.6/intel/lib/python3.8/site-packages/IPython/core/ultratb.py\", line 350, in _fixed_getinnerframes\n",
      "    records = fix_frame_records_filenames(inspect.getinnerframes(etb, context))\n",
      "  File \"/share/apps/python/3.8.6/intel/lib/python3.8/inspect.py\", line 1503, in getinnerframes\n",
      "    frameinfo = (tb.tb_frame,) + getframeinfo(tb, context)\n",
      "AttributeError: 'tuple' object has no attribute 'tb_frame'\n"
     ]
    },
    {
     "ename": "TypeError",
     "evalue": "object of type 'NoneType' has no len()",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mGetoptError\u001b[0m                               Traceback (most recent call last)",
      "\u001b[0;32m/share/apps/python/3.8.6/intel/lib/python3.8/distutils/fancy_getopt.py\u001b[0m in \u001b[0;36mgetopt\u001b[0;34m(self, args, object)\u001b[0m\n\u001b[1;32m    232\u001b[0m         \u001b[0;32mtry\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 233\u001b[0;31m             \u001b[0mopts\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0margs\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mgetopt\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mgetopt\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0margs\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mshort_opts\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mlong_opts\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    234\u001b[0m         \u001b[0;32mexcept\u001b[0m \u001b[0mgetopt\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0merror\u001b[0m \u001b[0;32mas\u001b[0m \u001b[0mmsg\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m/share/apps/python/3.8.6/intel/lib/python3.8/getopt.py\u001b[0m in \u001b[0;36mgetopt\u001b[0;34m(args, shortopts, longopts)\u001b[0m\n\u001b[1;32m     94\u001b[0m         \u001b[0;32melse\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m---> 95\u001b[0;31m             \u001b[0mopts\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0margs\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mdo_shorts\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mopts\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0margs\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0;36m0\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0;36m1\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mshortopts\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0margs\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0;36m1\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m     96\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m/share/apps/python/3.8.6/intel/lib/python3.8/getopt.py\u001b[0m in \u001b[0;36mdo_shorts\u001b[0;34m(opts, optstring, shortopts, args)\u001b[0m\n\u001b[1;32m    194\u001b[0m         \u001b[0mopt\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0moptstring\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0moptstring\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0;36m0\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0moptstring\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0;36m1\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 195\u001b[0;31m         \u001b[0;32mif\u001b[0m \u001b[0mshort_has_arg\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mopt\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mshortopts\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    196\u001b[0m             \u001b[0;32mif\u001b[0m \u001b[0moptstring\u001b[0m \u001b[0;34m==\u001b[0m \u001b[0;34m''\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m/share/apps/python/3.8.6/intel/lib/python3.8/getopt.py\u001b[0m in \u001b[0;36mshort_has_arg\u001b[0;34m(opt, shortopts)\u001b[0m\n\u001b[1;32m    210\u001b[0m             \u001b[0;32mreturn\u001b[0m \u001b[0mshortopts\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mstartswith\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m':'\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mi\u001b[0m\u001b[0;34m+\u001b[0m\u001b[0;36m1\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 211\u001b[0;31m     \u001b[0;32mraise\u001b[0m \u001b[0mGetoptError\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0m_\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m'option -%s not recognized'\u001b[0m\u001b[0;34m)\u001b[0m \u001b[0;34m%\u001b[0m \u001b[0mopt\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mopt\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    212\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;31mGetoptError\u001b[0m: option -f not recognized",
      "\nDuring handling of the above exception, another exception occurred:\n",
      "\u001b[0;31mDistutilsArgError\u001b[0m                         Traceback (most recent call last)",
      "\u001b[0;32m/share/apps/python/3.8.6/intel/lib/python3.8/distutils/core.py\u001b[0m in \u001b[0;36msetup\u001b[0;34m(**attrs)\u001b[0m\n\u001b[1;32m    133\u001b[0m     \u001b[0;32mtry\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 134\u001b[0;31m         \u001b[0mok\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mdist\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mparse_command_line\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    135\u001b[0m     \u001b[0;32mexcept\u001b[0m \u001b[0mDistutilsArgError\u001b[0m \u001b[0;32mas\u001b[0m \u001b[0mmsg\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m/share/apps/python/3.8.6/intel/lib/python3.8/distutils/dist.py\u001b[0m in \u001b[0;36mparse_command_line\u001b[0;34m(self)\u001b[0m\n\u001b[1;32m    474\u001b[0m         \u001b[0mparser\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mset_aliases\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m{\u001b[0m\u001b[0;34m'licence'\u001b[0m\u001b[0;34m:\u001b[0m \u001b[0;34m'license'\u001b[0m\u001b[0;34m}\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 475\u001b[0;31m         \u001b[0margs\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mparser\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mgetopt\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0margs\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mscript_args\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mobject\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    476\u001b[0m         \u001b[0moption_order\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mparser\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mget_option_order\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m/share/apps/python/3.8.6/intel/lib/python3.8/distutils/fancy_getopt.py\u001b[0m in \u001b[0;36mgetopt\u001b[0;34m(self, args, object)\u001b[0m\n\u001b[1;32m    234\u001b[0m         \u001b[0;32mexcept\u001b[0m \u001b[0mgetopt\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0merror\u001b[0m \u001b[0;32mas\u001b[0m \u001b[0mmsg\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 235\u001b[0;31m             \u001b[0;32mraise\u001b[0m \u001b[0mDistutilsArgError\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mmsg\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    236\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;31mDistutilsArgError\u001b[0m: option -f not recognized",
      "\nDuring handling of the above exception, another exception occurred:\n",
      "\u001b[0;31mSystemExit\u001b[0m                                Traceback (most recent call last)",
      "    \u001b[0;31m[... skipping hidden 1 frame]\u001b[0m\n",
      "\u001b[0;32m<ipython-input-1-52f8324f9f8e>\u001b[0m in \u001b[0;36m<module>\u001b[0;34m\u001b[0m\n\u001b[1;32m      7\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m----> 8\u001b[0;31m setup(name=\"submeso_ML\",\n\u001b[0m\u001b[1;32m      9\u001b[0m     \u001b[0mversion\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mversion\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m/share/apps/python/3.8.6/intel/lib/python3.8/site-packages/setuptools/__init__.py\u001b[0m in \u001b[0;36msetup\u001b[0;34m(**attrs)\u001b[0m\n\u001b[1;32m    164\u001b[0m     \u001b[0m_install_setup_requires\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mattrs\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 165\u001b[0;31m     \u001b[0;32mreturn\u001b[0m \u001b[0mdistutils\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mcore\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0msetup\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m**\u001b[0m\u001b[0mattrs\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    166\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m/share/apps/python/3.8.6/intel/lib/python3.8/distutils/core.py\u001b[0m in \u001b[0;36msetup\u001b[0;34m(**attrs)\u001b[0m\n\u001b[1;32m    135\u001b[0m     \u001b[0;32mexcept\u001b[0m \u001b[0mDistutilsArgError\u001b[0m \u001b[0;32mas\u001b[0m \u001b[0mmsg\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 136\u001b[0;31m         \u001b[0;32mraise\u001b[0m \u001b[0mSystemExit\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mgen_usage\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mdist\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mscript_name\u001b[0m\u001b[0;34m)\u001b[0m \u001b[0;34m+\u001b[0m \u001b[0;34m\"\\nerror: %s\"\u001b[0m \u001b[0;34m%\u001b[0m \u001b[0mmsg\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    137\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;31mSystemExit\u001b[0m: usage: ipykernel_launcher.py [global_opts] cmd1 [cmd1_opts] [cmd2 [cmd2_opts] ...]\n   or: ipykernel_launcher.py --help [cmd1 cmd2 ...]\n   or: ipykernel_launcher.py --help-commands\n   or: ipykernel_launcher.py cmd --help\n\nerror: option -f not recognized",
      "\nDuring handling of the above exception, another exception occurred:\n",
      "\u001b[0;31mTypeError\u001b[0m                                 Traceback (most recent call last)",
      "    \u001b[0;31m[... skipping hidden 1 frame]\u001b[0m\n",
      "\u001b[0;32m/share/apps/python/3.8.6/intel/lib/python3.8/site-packages/IPython/core/interactiveshell.py\u001b[0m in \u001b[0;36mshowtraceback\u001b[0;34m(self, exc_tuple, filename, tb_offset, exception_only, running_compiled_code)\u001b[0m\n\u001b[1;32m   2035\u001b[0m                     stb = ['An exception has occurred, use %tb to see '\n\u001b[1;32m   2036\u001b[0m                            'the full traceback.\\n']\n\u001b[0;32m-> 2037\u001b[0;31m                     stb.extend(self.InteractiveTB.get_exception_only(etype,\n\u001b[0m\u001b[1;32m   2038\u001b[0m                                                                      value))\n\u001b[1;32m   2039\u001b[0m                 \u001b[0;32melse\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m/share/apps/python/3.8.6/intel/lib/python3.8/site-packages/IPython/core/ultratb.py\u001b[0m in \u001b[0;36mget_exception_only\u001b[0;34m(self, etype, value)\u001b[0m\n\u001b[1;32m    821\u001b[0m         \u001b[0mvalue\u001b[0m \u001b[0;34m:\u001b[0m \u001b[0mexception\u001b[0m \u001b[0mvalue\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    822\u001b[0m         \"\"\"\n\u001b[0;32m--> 823\u001b[0;31m         \u001b[0;32mreturn\u001b[0m \u001b[0mListTB\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mstructured_traceback\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0metype\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mvalue\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    824\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    825\u001b[0m     \u001b[0;32mdef\u001b[0m \u001b[0mshow_exception_only\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0metype\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mevalue\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m/share/apps/python/3.8.6/intel/lib/python3.8/site-packages/IPython/core/ultratb.py\u001b[0m in \u001b[0;36mstructured_traceback\u001b[0;34m(self, etype, evalue, etb, tb_offset, context)\u001b[0m\n\u001b[1;32m    696\u001b[0m             \u001b[0mchained_exceptions_tb_offset\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0;36m0\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    697\u001b[0m             out_list = (\n\u001b[0;32m--> 698\u001b[0;31m                 self.structured_traceback(\n\u001b[0m\u001b[1;32m    699\u001b[0m                     \u001b[0metype\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mevalue\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m(\u001b[0m\u001b[0metb\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mchained_exc_ids\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    700\u001b[0m                     chained_exceptions_tb_offset, context)\n",
      "\u001b[0;32m/share/apps/python/3.8.6/intel/lib/python3.8/site-packages/IPython/core/ultratb.py\u001b[0m in \u001b[0;36mstructured_traceback\u001b[0;34m(self, etype, value, tb, tb_offset, number_of_lines_of_context)\u001b[0m\n\u001b[1;32m   1433\u001b[0m         \u001b[0;32melse\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   1434\u001b[0m             \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mtb\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mtb\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m-> 1435\u001b[0;31m         return FormattedTB.structured_traceback(\n\u001b[0m\u001b[1;32m   1436\u001b[0m             self, etype, value, tb, tb_offset, number_of_lines_of_context)\n\u001b[1;32m   1437\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m/share/apps/python/3.8.6/intel/lib/python3.8/site-packages/IPython/core/ultratb.py\u001b[0m in \u001b[0;36mstructured_traceback\u001b[0;34m(self, etype, value, tb, tb_offset, number_of_lines_of_context)\u001b[0m\n\u001b[1;32m   1333\u001b[0m         \u001b[0;32mif\u001b[0m \u001b[0mmode\u001b[0m \u001b[0;32min\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mverbose_modes\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   1334\u001b[0m             \u001b[0;31m# Verbose modes need a full traceback\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m-> 1335\u001b[0;31m             return VerboseTB.structured_traceback(\n\u001b[0m\u001b[1;32m   1336\u001b[0m                 \u001b[0mself\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0metype\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mvalue\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mtb\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mtb_offset\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mnumber_of_lines_of_context\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   1337\u001b[0m             )\n",
      "\u001b[0;32m/share/apps/python/3.8.6/intel/lib/python3.8/site-packages/IPython/core/ultratb.py\u001b[0m in \u001b[0;36mstructured_traceback\u001b[0;34m(self, etype, evalue, etb, tb_offset, number_of_lines_of_context)\u001b[0m\n\u001b[1;32m   1190\u001b[0m         \u001b[0;34m\"\"\"Return a nice text document describing the traceback.\"\"\"\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   1191\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m-> 1192\u001b[0;31m         formatted_exception = self.format_exception_as_a_whole(etype, evalue, etb, number_of_lines_of_context,\n\u001b[0m\u001b[1;32m   1193\u001b[0m                                                                tb_offset)\n\u001b[1;32m   1194\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m/share/apps/python/3.8.6/intel/lib/python3.8/site-packages/IPython/core/ultratb.py\u001b[0m in \u001b[0;36mformat_exception_as_a_whole\u001b[0;34m(self, etype, evalue, etb, number_of_lines_of_context, tb_offset)\u001b[0m\n\u001b[1;32m   1148\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   1149\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m-> 1150\u001b[0;31m         \u001b[0mlast_unique\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mrecursion_repeat\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mfind_recursion\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0morig_etype\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mevalue\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mrecords\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m   1151\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   1152\u001b[0m         \u001b[0mframes\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mformat_records\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mrecords\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mlast_unique\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mrecursion_repeat\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m/share/apps/python/3.8.6/intel/lib/python3.8/site-packages/IPython/core/ultratb.py\u001b[0m in \u001b[0;36mfind_recursion\u001b[0;34m(etype, value, records)\u001b[0m\n\u001b[1;32m    449\u001b[0m     \u001b[0;31m# first frame (from in to out) that looks different.\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    450\u001b[0m     \u001b[0;32mif\u001b[0m \u001b[0;32mnot\u001b[0m \u001b[0mis_recursion_error\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0metype\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mvalue\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mrecords\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 451\u001b[0;31m         \u001b[0;32mreturn\u001b[0m \u001b[0mlen\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mrecords\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;36m0\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    452\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    453\u001b[0m     \u001b[0;31m# Select filename, lineno, func_name to track frames with\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;31mTypeError\u001b[0m: object of type 'NoneType' has no len()"
     ]
    }
   ],
   "source": [
    "#!/usr/bin/env python\n",
    "\n",
    "from setuptools import setup, find_packages\n",
    "\n",
    "description = \"Wrapper for submeso_ML experiments\"\n",
    "version=\"0.0.1\"\n",
    "\n",
    "setup(name=\"submeso_ML\",\n",
    "    version=version,\n",
    "    description=description,\n",
    "    url=\"https://github.com/abodner/submeso_ML\",\n",
    "    author=\"Abigail Bodner\",\n",
    "    author_email=\"abigail_bodner@nyu.edu\",\n",
    "    packages=find_packages(),\n",
    "    )"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.6"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
