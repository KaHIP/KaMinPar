/*******************************************************************************
 * Composable label propagation building blocks.
 *
 * @file:   label_propagation.h
 ******************************************************************************/
#pragma once

#include "kaminpar-common/datastructures/concurrent_fast_reset_array.h"
#include "kaminpar-common/datastructures/dynamic_map.h"
#include "kaminpar-common/datastructures/rating_map.h"
#include "kaminpar-common/label_propagation/active_set.h"
#include "kaminpar-common/label_propagation/cluster_chooser.h"
#include "kaminpar-common/label_propagation/kernel.h"
#include "kaminpar-common/label_propagation/move.h"
#include "kaminpar-common/label_propagation/passes/growing_hash_tables.h"
#include "kaminpar-common/label_propagation/passes/single_phase.h"
#include "kaminpar-common/label_propagation/passes/two_phase.h"
#include "kaminpar-common/label_propagation/postprocessing.h"
#include "kaminpar-common/label_propagation/rating_accumulator.h"
#include "kaminpar-common/label_propagation/run.h"
#include "kaminpar-common/label_propagation/stores.h"
#include "kaminpar-common/label_propagation/types.h"
#include "kaminpar-common/label_propagation/workspace.h"
