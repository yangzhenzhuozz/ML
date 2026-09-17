import { defineConfig } from 'vitepress';
import { withSidebar } from 'vitepress-sidebar';
import katexPlugin from '@vscode/markdown-it-katex';
import { withMermaid } from 'vitepress-plugin-mermaid';

export default withMermaid({
    ...defineConfig(
        withSidebar(
            {
                title: '首页',
                themeConfig: {
                    search: {
                        provider: 'local',
                    },
                },
                markdown: {
                    breaks: true,
                    config: (md) => {
                        md.use((katexPlugin as any).default);
                    },
                },
                vite: {
                    // 👇 新增以下配置以解决 dayjs 导出错误
                    optimizeDeps: {
                        include: ['dayjs', 'mermaid'],
                    },
                    ssr: {
                        noExternal: ['dayjs', 'mermaid', 'vitepress-plugin-mermaid'],
                    },
                    // 👆 新增结束
                    plugins: [
                        {
                            name: 'perf-markdown-title-hmr',
                            configureServer(server) {
                                server.watcher.on('change', (file) => {
                                    if (file.endsWith('.md')) {
                                        const siteDataModule = server.moduleGraph.getModuleById('\0@siteData');
                                        if (siteDataModule) {
                                            server.moduleGraph.invalidateModule(siteDataModule);
                                            server.reloadModule(siteDataModule);
                                        }
                                    }
                                });
                            },
                        },
                    ],
                },
            },
            {
                documentRootPath: 'docs',
                collapsed: true,
                useFolderLinkFromIndexFile: true,
                useTitleFromFileHeading: true,
                useFolderTitleFromIndexFile: true,
                removePrefixAfterOrdering: true,
                prefixSeparator: '-',
            },
        ),
    ),
    base: '/ML/',
    title: '我的文档',
    mermaid: {
        //mermaidConfig !theme here works for ligth mode since dark theme is forced in dark mode
    },
});
