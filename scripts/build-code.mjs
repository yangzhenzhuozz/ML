// 构建前生成网页可下载的 torch-code.zip。
// 关键设计：zip 是【构建产物】，只出现在构建输出/发布分支里，不进 git 源码树（已被 .gitignore 排除），
// 因此 git 中对 Torch/*.ts 的 diff、history 完全不受影响，且每次构建都从源码重新生成，天然与源码同步。
import { execFileSync } from 'node:child_process';
import { mkdirSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = join(dirname(fileURLToPath(import.meta.url)), '..'); // 仓库根
const srcDir = join(root, 'examples', 'Torch');
const outDir = join(root, 'docs', 'public', 'code');
const files = ['torch.ts', 'demo-xor.ts', 'torch.test.ts'];
const zipPath = join(outDir, 'torch-code.zip');

mkdirSync(outDir, { recursive: true });

// 用 python3 生成 zip（本地与 GitHub Actions ubuntu runner 都自带），避免引入额外 npm 依赖
const py = [
    'import zipfile, os',
    `os.chdir(${JSON.stringify(srcDir)})`,
    `z = zipfile.ZipFile(${JSON.stringify(zipPath)}, 'w', zipfile.ZIP_DEFLATED)`,
    `for f in ${JSON.stringify(files)}: z.write(f)`,
    'z.close()',
].join('\n');

try {
    execFileSync('python3', ['-c', py], { stdio: 'pipe' });
    console.log(`[generate:code] 已生成 ${zipPath}`);
} catch {
    console.warn('[generate:code] python3 不可用，跳过 zip 生成（页面下载链接将暂不可用）');
}