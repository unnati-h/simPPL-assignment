import { useNavigate, Link } from "react-router-dom";

export default function Home() {
  const navigate = useNavigate();

  return (
    <div className="min-h-screen bg-black text-white flex flex-col">

      <header className="py-4 px-6 border-b border-neutral-800 flex items-center justify-between">
        <Link to="/" className="flex items-center gap-2">
          <span className="text-lg font-semibold tracking-tight">datafy</span>
        </Link>
        <nav className="text-sm text-neutral-400">
          <Link to="/chats" className="hover:text-white transition">Chats</Link>
        </nav>
      </header>


      <main className="flex-1 flex flex-col items-center justify-center text-center px-6">
        <h1 className="text-5xl md:text-7xl font-extrabold leading-tight mb-4">
          Ask. Analyze. <span className="text-neutral-400">datafy.</span>
        </h1>
        <p className="max-w-2xl text-neutral-300 mb-8">
        Welcome to datafy it answers all your questions about recent social media patterns and trends helping with research, data analysis, market research, trend analysis and other things
        </p>
        <button
          onClick={() => navigate("/chats")}
          className="bg-white text-black px-7 py-3 rounded-full font-semibold hover:bg-neutral-200 active:scale-95 transition"
          aria-label="Get started with datafy"
        >
          Get Started
        </button>
      </main>

      <section className="py-16 px-6 border-t border-neutral-800">
        <div className="max-w-5xl mx-auto">
          <h2 className="text-2xl font-semibold mb-6">How to use the app</h2>
          <ol className="list-decimal list-inside space-y-3 text-neutral-300">
            <li>Open the Chats page and ask a question (e.g., “Top 10 hashtags with a bar chart”).</li>
            <li>View the answer along with visualizations </li>
            <li> Download the PDF research paper style report for each chat</li>
          </ol>

          <div className="grid sm:grid-cols-3 gap-4 mt-10">
            <Feature title="Natural Q&A" desc="Ask in plain English; get precise, reproducible results." />
            <Feature title="Auto Charts" desc="Charts and Visualizations for easier analysis." />
            <Feature title="One-click PDF" desc="Compile findings into a research-style report." />
          </div>
        </div>
      </section>

      <footer className="py-6 text-center text-neutral-500 text-sm border-t border-neutral-800">
        © {new Date().getFullYear()} datafy
      </footer>
    </div>
  );
}

function Feature({ title, desc }) {
  return (
    <div className="rounded-2xl border border-neutral-800 p-4">
      <h3 className="font-semibold mb-1">{title}</h3>
      <p className="text-neutral-400 text-sm">{desc}</p>
    </div>
  );
}
