armspeech
=========

This software provides a framework and example experiments for investigation
into probabilistic modelling of speech for statistical speech synthesis.
There is a particular focus on autoregressive models.

It grew out of experiments with autoregressive acoustic models for the author's
PhD thesis, with the goal of allowing rapid prototyping of different models.
As such it has been designed with productivity and flexibility in mind rather
than runtime speed.
It is very much a work in progress.


Set-up
------

[armspeech](https://github.com/MattShannon/armspeech) is hosted on github.
To obtain the latest source code using git:

    git clone git://github.com/MattShannon/armspeech.git

Many of the formats used in armspeech are similar to those used in [HTS][hts].
In particular armspeech expects HTS-style speech parameter and label files,
for example as produced by the [HTS demo][hts_demo].
The default method for generating audio from the generated speech parameters
is to use the [STRAIGHT vocoder][straight].
By default the experiments use the [CMU ARCTIC corpus][arctic], speaker slt.

armspeech has the following dependencies:

- CMU ARCTIC corpus, processed into HTS-style speech parameter and label files
  (for example, by the HTS demo)
- if you want to generate audio, STRAIGHT vocoder (which requires MATLAB)
- if you want to generate audio, an appropriate HTS demo-style `Config.pm` file
- the [codedep][codedep] package for code-level dependency tracking
- python (>= 2.6) with recent numpy, scipy and matplotlib
- if using the HTS demo to generate the required files above (recommended),
  you should use the STRAIGHT version of the English speaker dependent training
  demo (which requires HTS, which in turn requires HTK).
  HTS 2.1 (for HTK 3.4) was used for testing.

To set-up this directory:

- add paths to an appropriate data directory and label directory in
  `expt_hts_demo/experiment.py` (by editing the strings starting '## TBA').
  The data directory should contain `.mgc`, `.lf0` and `.bap` files.
  The label directory should contain `.lab` files, each of which is an
  alignment with full-context labels.
  Either phone-level or state-level alignments may be used (but note that some
  of the example experiments require state-level alignments).
- update `mgcOrder` (two places) and `subLabels` (one place) in
  `expt_hts_demo/experiment.py` (where the corpus objects are created) to have
  values appropriate for your corpus.
- if you want to generate audio, add an appropriate `scripts/Config.pm` file
  (e.g. copied from the HTS demo)
- if necessary make `print_pickle.py`, `run_expt_hts_demo.sh` and `run_tests.sh`
  executable (use `chmod u+x`)

You can then run example experiments using:

    ./run_expt_hts_demo.sh

Currently `expt_hts_demo` uses the `armspeech` python package as a library, but
the latter is not intended to be a fully-fledged package suitable for separate
distribution.
This may change as the code matures.


License
-------

Please see the file `License` for details of the license and warranty for armspeech.

Parts of the code in this directory are based on the following software packages:

- [GPML toolbox][gpml] v3.0
- [HTS demo][hts_demo] (STRAIGHT version of the English speaker dependent training demo for HTS 2.1)


Bugs
----

Please use the [issue tracker](https://github.com/MattShannon/armspeech/issues)
to submit bug reports.


Contact
-------

The author of armspeech is [Matt Shannon](mailto:matt.shannon@cantab.net).


[hts]: http://hts.sp.nitech.ac.jp/ "HMM-based Speech Synthesis System (HTS)"
[hts_demo]: http://hts.sp.nitech.ac.jp/?Download
[straight]: http://www.wakayama-u.ac.jp/~kawahara/STRAIGHTadv/index_e.html
[arctic]: http://festvox.org/cmu_arctic/
[gpml]: http://www.gaussianprocess.org/gpml/code/matlab/doc/index.html
[codedep]: https://github.com/MattShannon/codedep
