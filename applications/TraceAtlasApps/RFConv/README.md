  Combining already written applications of sync_MMSE_beamformer (Matt), STAP_Comms (Matt), 
  and temporal_mitigation (Saquib) to make a full (simplistic) RF convergence simulation 
  in which both a radar and communications signal are overlapping in time and frequency.
  We sync to the comms signal, MMSE estimate, decode and remodulate, then temporally project 
  the full received data onto the subspace orthogonal to the received comms signal.
