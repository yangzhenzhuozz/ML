import { defineConfig } from 'vitepress';
import { withSidebar } from 'vitepress-sidebar';
import katexPlugin from '@vscode/markdown-it-katex';
export default defineConfig(
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
                plugins: [
                    {
                        name: 'perf-markdown-title-hmr',
                        configureServer(server) {
                            server.watcher.on('change', (file) => {
                                if (file.endsWith('.md')) {
                                    // VitePress 的路由和侧边栏数据挂载在 @siteData 这个虚拟模块上的
                                    const siteDataModule =
                                        server.moduleGraph.getModuleById(
                                            '\0@siteData',
                                        );
                                    if (siteDataModule) {
                                        server.moduleGraph.invalidateModule(
                                            siteDataModule,
                                        );
                                        server.reloadModule(siteDataModule); //只向浏览器发送这个单一数据的更新信号（局部热更新）
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
);
