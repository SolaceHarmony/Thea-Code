import { EXPERIMENT_IDS, experimentConfigsMap, experiments as Experiments, ExperimentId } from "../experiments"

import * as assert from 'assert'
suite("experiments", () => {
	suite("POWER_STEERING", () => {
		test("is configured correctly", () => {
			assert.strictEqual(EXPERIMENT_IDS.POWER_STEERING, "powerSteering")
			expect(experimentConfigsMap.POWER_STEERING).toMatchObject({
				enabled: false,

	suite("isEnabled", () => {
		test("returns false when experiment is not enabled", () => {
			const experiments: Partial<Record<ExperimentId, boolean>> = {
				powerSteering: false,
				experimentalDiffStrategy: false,
				search_and_replace: false,
				insert_content: false,

			expect(Experiments.isEnabled(experiments, EXPERIMENT_IDS.POWER_STEERING)).toBe(false)

		test("returns true when experiment is enabled", () => {
			const experiments: Partial<Record<ExperimentId, boolean>> = {
				powerSteering: true,
				experimentalDiffStrategy: false,
				search_and_replace: false,
				insert_content: false,

			expect(Experiments.isEnabled(experiments, EXPERIMENT_IDS.POWER_STEERING)).toBe(true)

		test("returns false when experiment is not present", () => {
			const experiments: Partial<Record<ExperimentId, boolean>> = {
				experimentalDiffStrategy: false,
				search_and_replace: false,
				insert_content: false,

			expect(Experiments.isEnabled(experiments, EXPERIMENT_IDS.POWER_STEERING)).toBe(false)
