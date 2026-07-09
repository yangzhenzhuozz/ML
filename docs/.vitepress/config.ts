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
        },
        {
            documentRootPath: 'docs',
            collapsed: true,
            useFolderLinkFromIndexFile: true,
            useTitleFromFileHeading: true,
            useFolderTitleFromIndexFile: true,
        },
    ),
);
