#!/usr/bin/env python
"""
langid.py - 
Language Identifier by Marco Lui April 2011

Based on research by Marco Lui and Tim Baldwin.

Copyright 2011 Marco Lui <saffsd@gmail.com>. All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are
permitted provided that the following conditions are met:

   1. Redistributions of source code must retain the above copyright notice, this list of
      conditions and the following disclaimer.

   2. Redistributions in binary form must reproduce the above copyright notice, this list
      of conditions and the following disclaimer in the documentation and/or other materials
      provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER ``AS IS'' AND ANY EXPRESS OR IMPLIED
WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

The views and conclusions contained in the software and documentation are those of the
authors and should not be interpreted as representing official policies, either expressed
or implied, of the copyright holder.
"""
from __future__ import print_function
from typing import List, Tuple, Optional, Union, Dict, Any
import array

NORM_PROBS = False # Normalize output probabilities.

# NORM_PROBS defaults to False for a small speed increase. It does not
# affect the relative ordering of the predicted classes. It can be 
# re-enabled at runtime - see the readme.

import base64
import bz2
import optparse
import sys
import logging
import numpy as np
import os
from collections import defaultdict


try:
  from cPickle import loads
except ImportError:
  from pickle import loads

logger = logging.getLogger(__name__)


# Convenience methods defined below will initialize this when first called.
identifier: Optional['LanguageIdentifier'] = None

def load_model(path: Optional[str] = None) -> None:
  """
  Load a model from binary files or use the default model.
  
  @param path path to the model directory, or None to use the default model
  """
  global identifier
  if path is None:
    path = "langid/data"
  
  if not os.path.exists(path):
    raise RuntimeError(f"Model directory not found: {path}")
  
  try:
    identifier = LanguageIdentifier.from_binary_files(path)
    logger.info("✅ Model loaded successfully from binary files!")
    logger.info(f"📁 Model directory: {path}/")
    logger.info("💾 Total size: ~7.3 MB (optimized format)")
    logger.info("🔧 Endianness: Little-endian")
  except Exception as e:
    logger.error(f"Failed to load model from {path}: {e}")
    raise RuntimeError(f"Failed to load model from {path}")

def set_languages(langs: Optional[List[str]] = None) -> 'LanguageIdentifier':
  """
  Set the language set used by the global identifier.

  @param langs a list of language codes
  @return the language identifier instance
  """
  global identifier
  if identifier is None:
    load_model()
    if identifier is None:
      raise RuntimeError("Failed to load model")
  return identifier.set_languages(langs)


def classify(instance: str) -> Tuple[str, float]:
  """
  Convenience method using a global identifier instance with the default
  model included in langid.py. Identifies the language that a string is 
  written in.

  @param instance a text string. Unicode strings will automatically be utf8-encoded
  @returns a tuple of the most likely language and the confidence score
  """
  global identifier
  if identifier is None:
    load_model()
    if identifier is None:
      raise RuntimeError("Failed to load model")
  return identifier.classify(instance)

def rank(instance: str) -> List[Tuple[str, float]]:
  """
  Convenience method using a global identifier instance with the default
  model included in langid.py. Ranks all the languages in the model according
  to the likelihood that the string is written in each language.

  @param instance a text string. Unicode strings will automatically be utf8-encoded
  @returns a list of tuples language and the confidence score, in descending order
  """
  global identifier
  if identifier is None:
    load_model()
    if identifier is None:
      raise RuntimeError("Failed to load model")
  return identifier.rank(instance)
  
def cl_path(path):
  """
  Convenience method using a global identifier instance with the default
  model included in langid.py. Identifies the language that the file at `path` is 
  written in.

  @param path path to file
  @returns a tuple of the most likely language and the confidence score
  """
  global identifier
  if identifier is None:
    load_model()

  return identifier.cl_path(path)

def rank_path(path):
  """
  Convenience method using a global identifier instance with the default
  model included in langid.py. Ranks all the languages in the model according
  to the likelihood that the file at `path` is written in each language.

  @param path path to file
  @returns a list of tuples language and the confidence score, in descending order
  """
  global identifier
  if identifier is None:
    load_model()

  return identifier.rank_path(path)


class LanguageIdentifier(object):
  """
  This class implements the actual language identifier.
  """

  @classmethod
  def from_modelstring(cls, string: Union[str, bytes], *args: Any, **kwargs: Any) -> 'LanguageIdentifier':
    """
    Create a LanguageIdentifier from a model string.
    
    Data flow:
    1. Decode base64 string -> compressed bytes
    2. Decompress bytes -> pickle data
    3. Unpickle data -> 5 model variables (List[float], List[float], List[str], array.array, Dict)
    4. Convert lists to numpy arrays -> (np.ndarray, np.ndarray, List[str], array.array, Dict)
    
    @param string the model string (base64 encoded and compressed)
    @param args additional arguments to pass to the constructor
    @param kwargs additional keyword arguments to pass to the constructor
    @return a new LanguageIdentifier instance
    """
    b = base64.b64decode(string)
    z = bz2.decompress(b)
    model = loads(z)
    
    # Type hints for the 5 model variables from pickle:
    # nb_ptc_list: List[float] - Probability table for features given classes (flattened)
    #   Shape: [7480 * 97] = 725,560 elements (7480 features × 97 languages)
    #   Range: -17.31 to -0.90 (log probabilities, all negative)
    # nb_pc_list: List[float] - Prior probabilities for each class  
    #   Shape: [97] = 97 elements (one per language)
    #   Range: 1.95 to 9.06 (log probabilities, all positive)
    # nb_classes: List[str] - List of language codes (e.g., ['en', 'fr', 'de'])
    #   Shape: [97] = 97 elements (one per language)
    #   Format: 2-letter ISO codes (e.g., 'en', 'fr', 'de')
    # tk_nextmove: array.array - Tokenizer state transitions (array of unsigned shorts, typecode 'H')
    #   Shape: [2,334,208] = 2,334,208 elements (state transition table)
    #   Range: 0 to 9,117 (state indices, 29% zeros)
    # tk_output: Dict[int, Tuple] - Tokenizer output mappings (state -> output tuple)
    #   Shape: 8,656 entries (state -> output mapping)
    #   Keys: 0 to 9,117 (state indices)
    #   Values: Empty tuples () or small tuples of uint16
    nb_ptc_list: List[float]
    nb_pc_list: List[float] 
    nb_classes: List[str]
    tk_nextmove: array.array
    tk_output: Dict[int, Tuple]
    nb_ptc_list, nb_pc_list, nb_classes, tk_nextmove, tk_output = model
    
    nb_numfeats = int(len(nb_ptc_list) / len(nb_pc_list))

    # Convert lists to numpy arrays
    nb_pc = np.array(nb_pc_list)  # Shape: [97] -> (97,)
    nb_ptc = np.array(nb_ptc_list).reshape(nb_numfeats, len(nb_pc))  # Shape: [725,560] -> (7480, 97)
    
    # Go Storage Recommendations:
    # - nb_ptc: [][]float32 (7480×97 matrix, range -17.31 to -0.90)
    # - nb_pc: []float32 (97 vector, range 1.95 to 9.06)
    # - nb_classes: []string (97 codes, 2-letter ISO format)
    # - tk_nextmove: []uint16 (2,334,208 elements, range 0-9,117, 29% zeros)
    # - tk_output: map[uint32][]uint16 (8,656 entries, keys 0-9,117)
    # Total optimized size: ~7.2MB (vs 10.0MB with float64)
   
    return cls(nb_ptc, nb_pc, nb_numfeats, nb_classes, tk_nextmove, tk_output, *args, **kwargs)

  @classmethod
  def from_binary_files(cls, model_dir: str, *args: Any, **kwargs: Any) -> 'LanguageIdentifier':
    """
    Create a LanguageIdentifier from binary files in the specified directory.
    
    Binary file format (all little-endian):
    - nb_ptc.bin: [rows:uint32][cols:uint32][data:float32[]] (7480×97 matrix)
    - nb_pc.bin: [length:uint32][data:float32[]] (97 vector)
    - nb_classes.json: [string, string, ...] (97 language codes)
    - tk_nextmove.bin: [length:uint32][data:uint16[]] (2,334,208 elements)
    - tk_output.bin: [num_entries:uint32][key:uint32][value_length:uint32][values:uint16[]]... (8,656 entries)
    
    @param model_dir path to directory containing the binary model files
    @param args additional arguments to pass to the constructor
    @param kwargs additional keyword arguments to pass to the constructor
    @return a new LanguageIdentifier instance
    """
    import struct
    import json
    
    logger.info("1. Loading nb_ptc (probability table)...")
    with open(os.path.join(model_dir, "nb_ptc.bin"), "rb") as f:
      rows, cols = struct.unpack("<II", f.read(8))  # Read shape (little-endian)
      nb_ptc = np.frombuffer(f.read(), dtype=np.float32).reshape(rows, cols)
    logger.info(f"   Loaded: {rows}×{cols} matrix, {nb_ptc.nbytes} bytes")
    
    logger.info("2. Loading nb_pc (prior probabilities)...")
    with open(os.path.join(model_dir, "nb_pc.bin"), "rb") as f:
      length = struct.unpack("<I", f.read(4))[0]  # Read length (little-endian)
      nb_pc = np.frombuffer(f.read(), dtype=np.float32)
    logger.info(f"   Loaded: {length} elements, {nb_pc.nbytes} bytes")
    
    logger.info("3. Loading nb_classes (language codes)...")
    with open(os.path.join(model_dir, "nb_classes.json"), "r") as f:
      nb_classes = json.load(f)
    logger.info(f"   Loaded: {len(nb_classes)} language codes")
    
    logger.info("4. Loading tk_nextmove (state transitions)...")
    with open(os.path.join(model_dir, "tk_nextmove.bin"), "rb") as f:
      length = struct.unpack("<I", f.read(4))[0]  # Read length (little-endian)
      tk_nextmove = array.array('H', f.read())  # Read as uint16 array
    logger.info(f"   Loaded: {length} elements, {tk_nextmove.buffer_info()[1]} bytes")
    
    logger.info("5. Loading tk_output (output mappings)...")
    with open(os.path.join(model_dir, "tk_output.bin"), "rb") as f:
      num_entries = struct.unpack("<I", f.read(4))[0]  # Read number of entries
      tk_output = {}
      for _ in range(num_entries):
        key = struct.unpack("<I", f.read(4))[0]  # Read key (uint32)
        value_length = struct.unpack("<I", f.read(4))[0]  # Read value length (uint32)
        values = array.array('H', f.read(value_length * 2))  # Read values (uint16[])
        tk_output[key] = tuple(values)
    logger.info(f"   Loaded: {num_entries} entries")
    
    # Calculate nb_numfeats from the loaded data
    nb_numfeats = nb_ptc.shape[0]  # Number of features (7480)
    
    return cls(nb_ptc, nb_pc, nb_numfeats, nb_classes, tk_nextmove, tk_output, *args, **kwargs)

  @classmethod
  def from_modelpath(cls, path: str, *args: Any, **kwargs: Any) -> Optional['LanguageIdentifier']:
    """
    Create a LanguageIdentifier from a model file.
    
    @param path path to the model file
    @param args additional arguments to pass to the constructor
    @param kwargs additional keyword arguments to pass to the constructor
    @return a new LanguageIdentifier instance
    """
    with open(path) as f:
      return cls.from_modelstring(f.read().encode(), *args, **kwargs)

  def __init__(self, nb_ptc: np.ndarray, nb_pc: np.ndarray, nb_numfeats: int, 
               nb_classes: List[str], tk_nextmove: array.array, tk_output: Dict[int, Tuple],
               norm_probs: bool = NORM_PROBS) -> None:
    """
    Initialize LanguageIdentifier with model data.
    
    @param nb_ptc: numpy.ndarray - Probability table for features given classes (reshaped)
      Shape: (7480, 97) - 7480 features × 97 languages
      Range: -17.31 to -0.90 (log probabilities, all negative)
    @param nb_pc: numpy.ndarray - Prior probabilities for each class
      Shape: (97,) - 97 languages
      Range: 1.95 to 9.06 (log probabilities, all positive)
    @param nb_numfeats: int - Number of features (7480)
    @param nb_classes: List[str] - List of language codes
      Shape: [97] - 97 languages
      Format: 2-letter ISO codes (e.g., 'en', 'fr', 'de')
    @param tk_nextmove: array.array - Tokenizer state transitions (typecode 'H')
      Shape: [2,334,208] - state transition table
      Range: 0 to 9,117 (state indices, 29% zeros)
    @param tk_output: Dict[int, Tuple] - Tokenizer output mappings
      Shape: 8,656 entries - state -> output mapping
      Keys: 0 to 9,117 (state indices)
      Values: Empty tuples () or small tuples of uint16
    @param norm_probs: bool - Whether to normalize probabilities
    """
    self.nb_ptc = nb_ptc
    self.nb_pc = nb_pc
    self.nb_numfeats = nb_numfeats
    self.nb_classes = nb_classes
    self.tk_nextmove = tk_nextmove
    self.tk_output = tk_output

    if norm_probs:
      def norm_probs(pd):
        """
        Renormalize log-probs into a proper distribution (sum 1)
        The technique for dealing with underflow is described in
        http://jblevins.org/log/log-sum-exp
        """
        # Ignore overflow when computing the exponential. Large values
        # in the exp produce a result of inf, which does not affect
        # the correctness of the calculation (as 1/x->0 as x->inf). 
        # On Linux this does not actually trigger a warning, but on 
        # Windows this causes a RuntimeWarning, so we explicitly 
        # suppress it.
        with np.errstate(over='ignore'):
          pd_exp = np.exp(pd)
          pd = pd_exp / pd_exp.sum()
        return pd
    else:
      def norm_probs(pd):
        return pd

    self.norm_probs = norm_probs

    # Maintain a reference to the full model, in case we change our language set
    # multiple times.
    self.__full_model = nb_ptc, nb_pc, nb_classes

  def set_languages(self, langs: Optional[List[str]] = None) -> 'LanguageIdentifier':
    """
    Set the languages to classify between.
    
    @param langs list of language codes to restrict to, or None for all languages
    @return self for method chaining
    """
    logger.debug("restricting languages to: %s", langs)

    # Unpack the full original model. This is needed in case the language set
    # has been previously trimmed, and the new set is not a subset of the current
    # set.
    nb_ptc, nb_pc, nb_classes = self.__full_model

    if langs is None:
      self.nb_classes = nb_classes 
      self.nb_ptc = nb_ptc
      self.nb_pc = nb_pc

    else:
      # We were passed a restricted set of languages. Trim the arrays accordingly
      # to speed up processing.
      for lang in langs:
        if lang not in nb_classes:
          raise ValueError("Unknown language code %s" % lang)

      subset_mask = np.fromiter((l in langs for l in nb_classes), dtype=bool)
      self.nb_classes = [ c for c in nb_classes if c in langs ]
      self.nb_ptc = nb_ptc[:,subset_mask]
      self.nb_pc = nb_pc[subset_mask]
    
    return self

  def instance2fv(self, text: str) -> np.ndarray:
    """
    Map an instance into the feature space of the trained model.
    
    @param text the text to convert to feature vector
    @return feature vector as numpy array
    """
    # Python3
    if isinstance(text,str):
      text = text.encode('utf8')

    arr = np.zeros((self.nb_numfeats,), dtype='uint32')

    # Count the number of times we enter each state
    state = 0
    statecount = defaultdict(int)
    for letter in text:
      state = self.tk_nextmove[(state << 8) + letter]
      statecount[state] += 1

    # Update all the productions corresponding to the state
    for state in statecount:
      for index in self.tk_output.get(state, []):
        arr[index] += statecount[state]

    return arr

  def nb_classprobs(self, fv: np.ndarray) -> np.ndarray:
    """
    Compute class probabilities for a feature vector.
    
    @param fv feature vector
    @return array of class probabilities
    """
    # compute the partial log-probability of the document given each class
    pdc = np.dot(fv,self.nb_ptc)
    # compute the partial log-probability of the document in each class
    pd = pdc + self.nb_pc
    return pd

  def classify(self, text: str) -> Tuple[str, float]:
    """
    Classify an instance.
    
    @param text the text to classify
    @return tuple of (language_code, confidence_score)
    """
    fv = self.instance2fv(text)
    probs = self.norm_probs(self.nb_classprobs(fv))  # type: ignore
    cl = np.argmax(probs)
    conf = float(probs[cl])
    pred = str(self.nb_classes[cl])
    return pred, conf

  def rank(self, text: str) -> List[Tuple[str, float]]:
    """
    Rank all languages by probability for the given text.
    
    @param text the text to classify
    @return list of (language_code, confidence_score) tuples, sorted by confidence
    """
    fv = self.instance2fv(text)
    probs = self.norm_probs(self.nb_classprobs(fv))  # type: ignore
    
    # Create list of (language, probability) tuples
    lang_probs = [(self.nb_classes[i], float(probs[i])) for i in range(len(self.nb_classes))]
    
    # Sort by probability in descending order
    lang_probs.sort(key=lambda x: x[1], reverse=True)
    
    return lang_probs



  def cl_path(self, path):
    """
    Classify a file at a given path
    """
    with open(path) as f:
      retval = self.classify(f.read())
    return path, retval

  def rank_path(self, path):
    """
    Class ranking for a file at a given path
    """
    with open(path) as f:
      retval = self.rank(f.read())
    return path, retval
      


def main():
  global identifier

  parser = optparse.OptionParser()
  parser.add_option('-m', dest='model', help='load model from file')
  parser.add_option('-l', '--langs', dest='langs', help='comma-separated set of target ISO639 language codes (e.g en,de)')
  parser.add_option('-d', '--dist', action='store_true', default=False, help='show full distribution over languages')
  parser.add_option('-n', '--normalize', action='store_true', default=False, help='normalize confidence scores to probability values')
  options, args = parser.parse_args()

  # unpack a model 
  if options.model:
    try:
      identifier = LanguageIdentifier.from_modelpath(options.model, norm_probs = options.normalize)
      logger.info("Using external model: %s", options.model)
    except IOError as e:
      logger.warning("Failed to load %s: %s" % (options.model,e))
  
  if identifier is None:
    identifier = LanguageIdentifier.from_modelstring(model, norm_probs = options.normalize)
    logger.info("Using internal model")

  if options.langs:
    langs = options.langs.split(",")
    identifier.set_languages(langs)

  def _process(text: str):
    """
    Set up a local function to do output, configured according to our settings.
    """
    if options.dist:
      payload = identifier.rank(text)
    else:
      payload = identifier.classify(text)

    return payload


 
  import sys
  if sys.stdin.isatty():
    # Interactive mode
    while True:
      try:
        print(">>>", end=' ')
        text = input()
      except Exception as e:
        print(e)
        break
      print(_process(text))
  else:
    # Redirected
    if options.line:
      for line in sys.stdin:
        print(_process(line))
    else:
      print(_process(sys.stdin.read()))
     

if __name__ == "__main__":
  main()
