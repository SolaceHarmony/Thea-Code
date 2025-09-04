import { strict as assert } from "node:assert"
import { EXPERIMENT_IDS, experimentConfigsMap, experiments as Experiments, ExperimentId } from "../experiments"

describe("experiments", () => {
	describe("POWER_STEERING", () => {
		it("is configured correctly", () => {
			assert.equal(EXPERIMENT_IDS.POWER_STEERING, "powerSteering")
			// explicit check instead of deep include to satisfy strict typing
			assert.equal((experimentConfigsMap.POWER_STEERING as { enabled?: boolean }).enabled, false)
		})
	})

	describe("isEnabled", () => {
		it("returns false when experiment is not enabled", () => {
			const experiments: Partial<Record<ExperimentId, boolean>> = {
				powerSteering: false,
				experimentalDiffStrategy: false,
				search_and_replace: false,
				insert_content: false,
			}
			assert.equal(Experiments.isEnabled(experiments, EXPERIMENT_IDS.POWER_STEERING), false)
		})

		it("returns true when experiment is enabled", () => {
			const experiments: Partial<Record<ExperimentId, boolean>> = {
				powerSteering: true,
				experimentalDiffStrategy: false,
				search_and_replace: false,
				insert_content: false,
			}
			assert.equal(Experiments.isEnabled(experiments, EXPERIMENT_IDS.POWER_STEERING), true)
		})

		it("returns false when experiment is not present", () => {
			const experiments: Partial<Record<ExperimentId, boolean>> = {
				experimentalDiffStrategy: false,
				search_and_replace: false,
				insert_content: false,
			}
			assert.equal(Experiments.isEnabled(experiments, EXPERIMENT_IDS.POWER_STEERING), false)
		})
	})
})
