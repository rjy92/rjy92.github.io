import { defineConfig } from 'vitepress'
import { withMermaid } from 'vitepress-plugin-mermaid'

export default withMermaid(
  defineConfig({
    lang: 'zh-CN',
    title: '我的网站',
    description: '前端开发者的文档、博客与学习笔记',

    // Mermaid 全局配置：统一字体避免测量与渲染不一致导致文字被裁切
    mermaid: {
      fontFamily: '"PingFang SC", "Microsoft YaHei", "Noto Sans SC", sans-serif',
      flowchart: { htmlLabels: true, padding: 10 }
    },

    // 顶部导航菜单
    themeConfig: {
      nav: [
        { text: '首页', link: '/' },
        { text: '博客', link: '/blog/' },
        { text: '学习', link: '/learn/' },
        { text: '关于', link: '/about' }
      ],

      // 侧边栏：按板块配置
      sidebar: {
        '/blog/': [
          {
            text: '博客',
            items: [
              { text: '全部文章', link: '/blog/' }
            ]
          },
          {
            text: 'AI',
            collapsed: false,
            items: [
              { text: '前端也能搞懂 RAG', link: '/blog/ai/frontend-rag' },
              { text: '模型缓存策略详解', link: '/blog/ai/model-cache-strategy' }
            ]
          }
        ],
        '/learn/': [
          {
            text: '学习',
            items: [
              { text: '总览', link: '/learn/' }
            ]
          }
        ]
      },

      socialLinks: [
        { icon: 'github', link: 'https://github.com/rjy92' }
      ],

      footer: {
        message: '基于 VitePress 构建',
        copyright: 'Copyright © 2026'
      },

      // 中文化
      outline: { level: [2, 3], label: '本页目录' },
      docFooter: { prev: '上一篇', next: '下一篇' },
      lastUpdated: { text: '最后更新' },
      returnToTopLabel: '返回顶部',
      darkModeSwitchLabel: '主题',
      sidebarMenuLabel: '菜单'
    }
  })
)
