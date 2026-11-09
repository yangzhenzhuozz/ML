// 构建前自动扫描 examples/ 下的每个一级目录，各自打成一个可下载的 zip。
// 关键设计：zip 是【构建产物】，只出现在构建输出/发布分支里，不进 git 源码树（已被 .gitignore 排除），
// 因此 git 中对源码的 diff、history 完全不受影响，且每次构建都从源码重新生成，天然与源码同步。
import { execFileSync } from 'node:child_process';
import { mkdirSync, readdirSync, rmSync } from 'node:fs';
import { dirname, join, relative } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = join(dirname(fileURLToPath(import.meta.url)), '..'); // 仓库根
const examplesDir = join(root, 'examples');
const outDir = join(root, 'docs', 'public', 'code');

// 目录名转 zip 名：去掉数字前缀、转小写、非字母数字转连字符，例如 08-Torch -> torch
function toSlug(name) {
    return name
        .replace(/^\d+[-_.\s]*/, '')
        .toLowerCase()
        .replace(/[^a-z0-9]+/g, '-')
        .replace(/^-+|-+$/g, '');
}

// 递归收集目录内所有普通文件（返回相对 base 的路径，保留子目录结构）
function collectFiles(dir, base = dir) {
    const files = [];
    for (const entry of readdirSync(dir, { withFileTypes: true })) {
        if (entry.name === 'node_modules' || entry.name === '.git') continue;
        if (entry.name.startsWith('.')) continue; // 跳过隐藏文件/目录
        const full = join(dir, entry.name);
        if (entry.isDirectory()) {
            files.push(...collectFiles(full, base));
        } else if (entry.isFile() && !entry.name.endsWith('.zip')) {
            files.push(relative(base, full));
        }
    }
    return files;
}

mkdirSync(outDir, { recursive: true });

// 清理上一次构建的产物，避免已删除目录残留旧 zip
for (const file of readdirSync(outDir)) {
    if (file.endsWith('-code.zip')) rmSync(join(outDir, file), { force: true });
}

// 用 python3 生成 zip（本地与 GitHub Actions ubuntu runner 都自带），避免引入额外 npm 依赖
const py = [
    'import os, sys, json, zipfile',
    'os.chdir(sys.argv[1])',
    'z = zipfile.ZipFile(sys.argv[2], "w", zipfile.ZIP_DEFLATED)',
    'for f in json.loads(sys.argv[3]): z.write(f)',
    'z.close()',
].join('\n');

const dirs = readdirSync(examplesDir, { withFileTypes: true })
    .filter((entry) => entry.isDirectory() && !entry.name.startsWith('.'))
    .map((entry) => entry.name)
    .sort();

let generated = 0;
for (const dirName of dirs) {
    const srcDir = join(examplesDir, dirName);
    const files = collectFiles(srcDir);
    if (files.length === 0) {
        console.warn(`[generate:code] 跳过空目录 ${dirName}`);
        continue;
    }

    const slug = toSlug(dirName);
    const zipName = slug ? `${slug}-code.zip` : `${dirName}-code.zip`;
    const zipPath = join(outDir, zipName);

    try {
        execFileSync('python3', ['-c', py, srcDir, zipPath, JSON.stringify(files)], { stdio: 'pipe' });
        console.log(`[generate:code] 已生成 ${join('code', zipName)}（${files.length} 个文件）`);
        generated++;
    } catch {
        console.warn(`[generate:code] python3 不可用，跳过 ${dirName} 的 zip 生成（页面下载链接将暂不可用）`);
    }
}

console.log(`[generate:code] 完成，共生成 ${generated} 个 zip`);