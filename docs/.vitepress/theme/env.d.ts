declare module '*.css' {}
declare module '*?worker' {
    const workerCtor: new (options?: WorkerOptions) => Worker;
    export default workerCtor;
}
