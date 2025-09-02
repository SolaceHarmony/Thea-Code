// Mock implementation for ShadowCheckpointService
import EventEmitter from "events"
import { CheckpointStorage } from "../../../shared/checkpoints"
import sinon from 'sinon'

export abstract class ShadowCheckpointService extends EventEmitter {
	public static getTaskStorage = sinon.stub().callsFake((): Promise<CheckpointStorage | undefined> => Promise.resolve('task'))

	// Use Sinon stubs for these methods
	public static deleteTask = sinon.stub().resolves(undefined)
	public static deleteBranch = sinon.stub().resolves(true)
	public static hashWorkspaceDir = sinon.stub().returns('mock-hash')
}
