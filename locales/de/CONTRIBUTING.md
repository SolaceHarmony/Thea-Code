# Beitrag zu Thea Code

Wir freuen uns, dass du Interesse hast, zu Thea Code beizutragen. Ob du einen Fehler behebst, eine Funktion hinzufügst oder unsere Dokumentation verbesserst, jeder Beitrag macht Thea Code intelligenter! Um unsere Community lebendig und einladend zu halten, müssen sich alle Mitglieder an unseren [Verhaltenskodex](CODE_OF_CONDUCT.md) halten.

## Treten Sie unserer Community bei

Wir ermutigen alle Mitwirkenden nachdrücklich, unserer [Discord-Community](https://discord.gg/thea-placeholder) beizutreten! Teil unseres Discord-Servers zu sein, hilft dir:

- Echtzeit-Hilfe und Anleitung für deine Beiträge zu erhalten
- Mit anderen Mitwirkenden und Kernteammitgliedern in Kontakt zu treten
- Über Projektentwicklungen und Prioritäten auf dem Laufenden zu bleiben
- An Diskussionen teilzunehmen, die die Zukunft von Thea Code gestalten
- Kooperationsmöglichkeiten mit anderen Entwicklern zu finden

## Fehler oder Probleme melden

Fehlerberichte helfen, Thea Code für alle besser zu machen! Bevor du ein neues Issue erstellst, bitte [suche in bestehenden Issues](SolaceHarmony/Thea-Code/issues), um Duplikate zu vermeiden. Wenn du bereit bist, einen Fehler zu melden, gehe zu unserer [Issues-Seite](SolaceHarmony/Thea-Code/issues/new/choose), wo du eine Vorlage findest, die dir beim Ausfüllen der relevanten Informationen hilft.

<blockquote class='warning-note'>
     🔐 <b>Wichtig:</b> Wenn du eine Sicherheitslücke entdeckst, nutze bitte das <a href="https://github.com/SolaceHarmony/Thea-Code/security/advisories/new">Github-Sicherheitstool, um sie privat zu melden</a>.
</blockquote>

## Entscheiden, woran Sie arbeiten möchten

Suchst du nach einem guten ersten Beitrag? Schau dir Issues im Abschnitt "Issue [Unassigned]" unseres [Thea Code Issues](https://github.com/orgs/sydneyrenee/projects/1) Github-Projekts an. Diese sind speziell für neue Mitwirkende und Bereiche ausgewählt, in denen wir Hilfe gebrauchen könnten!

Wir begrüßen auch Beiträge zu unserer [Dokumentation](https://docs.thea-placeholder.com/)! Ob du Tippfehler korrigierst, bestehende Anleitungen verbesserst oder neue Bildungsinhalte erstellst - wir würden gerne ein Community-geführtes Repository von Ressourcen aufbauen, das jedem hilft, das Beste aus Thea Code herauszuholen. Du kannst auf jeder Seite auf "Edit this page" klicken, um schnell zur richtigen Stelle in Github zu gelangen, um die Datei zu bearbeiten, oder du kannst direkt zu SolaceHarmony/Thea-Code-Docs gehen.

Wenn du an einer größeren Funktion arbeiten möchtest, erstelle bitte zuerst eine [Funktionsanfrage](SolaceHarmony/Thea-Code/discussions/categories/feature-requests?discussions_q=is%3Aopen+category%3A%22Feature+Requests%22+sort%3Atop), damit wir diskutieren können, ob sie mit der Vision von Thea Code übereinstimmt. Du kannst auch unseren [Projekt-Fahrplan](#projekt-fahrplan) unten überprüfen, um zu sehen, ob deine Idee mit unserer strategischen Ausrichtung übereinstimmt.

## Projekt-Fahrplan

Thea Code hat einen klaren Entwicklungsfahrplan, der unsere Prioritäten und zukünftige Richtung leitet. Das Verständnis unseres Fahrplans kann dir helfen:

- Deine Beiträge mit den Projektzielen abzustimmen
- Bereiche zu identifizieren, in denen deine Expertise am wertvollsten wäre
- Den Kontext hinter bestimmten Designentscheidungen zu verstehen
- Inspiration für neue Funktionen zu finden, die unsere Vision unterstützen

Unser aktueller Fahrplan konzentriert sich auf sechs Schlüsselsäulen:

### Provider-Unterstützung

Wir möchten so viele Provider wie möglich gut unterstützen:

- Vielseitigere "OpenAI Compatible" Unterstützung
- xAI, Microsoft Azure AI, Alibaba Cloud Qwen, IBM Watsonx, Together AI, DeepInfra, Fireworks AI, Cohere, Perplexity AI, FriendliAI, Replicate
- Verbesserte Unterstützung für Ollama und LM Studio

### Modell-Unterstützung

Wir wollen, dass Thea mit so vielen Modellen wie möglich gut funktioniert, einschließlich lokaler Modelle:

- Lokale Modellunterstützung durch benutzerdefiniertes System-Prompting und Workflows
- Benchmark-Evaluierungen und Testfälle

### System-Unterstützung

Wir wollen, dass Thea auf jedem Computer gut läuft:

- Plattformübergreifende Terminal-Integration
- Starke und konsistente Unterstützung für Mac, Windows und Linux

### Dokumentation

Wir wollen umfassende, zugängliche Dokumentation für alle Benutzer und Mitwirkenden:

- Erweiterte Benutzerhandbücher und Tutorials
- Klare API-Dokumentation
- Bessere Anleitung für Mitwirkende
- Mehrsprachige Dokumentationsressourcen
- Interaktive Beispiele und Codebeispiele

### Stabilität

Wir wollen die Anzahl der Fehler deutlich reduzieren und die automatisierte Testabdeckung erhöhen:

- Debug-Logging-Schalter
- "Maschinen-/Aufgabeninformationen" Kopier-Button zum Einsenden mit Fehler-/Support-Anfragen

### Internationalisierung

Wir wollen, dass Thea die Sprache aller spricht:

- 我们希望 Thea Code 说每个人的语言
- Queremos que Thea Code hable el idioma de todos
- हम चाहते हैं कि Thea Code हर किसी की भाषा बोले
- نريد أن يتحدث Thea Code لغة الجميع

Wir begrüßen besonders Beiträge, die unsere Fahrplanziele voranbringen. Wenn du an etwas arbeitest, das mit diesen Säulen übereinstimmt, erwähne es bitte in deiner PR-Beschreibung.

## Entwicklungs-Setup

1. **Klone** das Repository:

```sh
git clone https://github.com/SolaceHarmony/Thea-Code.git
```

1. **Installiere Abhängigkeiten**:

```sh
npm run install:all
```

1. **Starte die Webansicht (Vite/React-App mit HMR)**:

```sh
npm run dev
```

1. **Debugging**:
   Drücke `F5` (oder **Ausführen** → **Debugging starten**) in VSCode, um eine neue Sitzung mit geladenem Thea Code zu öffnen.

Änderungen an der Webansicht erscheinen sofort. Änderungen an der Kern-Erweiterung erfordern einen Neustart des Erweiterungs-Hosts.

Alternativ kannst du eine .vsix-Datei erstellen und direkt in VSCode installieren:

```sh
npm run build
```

Eine `.vsix`-Datei erscheint im `bin/`-Verzeichnis, die mit folgendem Befehl installiert werden kann:

```sh
code --install-extension bin/thea-code-<version>.vsix
```

## Code schreiben und einreichen

Jeder kann Code zu Thea Code beitragen, aber wir bitten dich, diese Richtlinien zu befolgen, um sicherzustellen, dass deine Beiträge reibungslos integriert werden können:

1. **Halten Sie Pull Requests fokussiert**

    - Beschränke PRs auf eine einzelne Funktion oder Fehlerbehebung
    - Teile größere Änderungen in kleinere, zusammenhängende PRs auf
    - Unterteile Änderungen in logische Commits, die unabhängig überprüft werden können

2. **Codequalität**

    - Alle PRs müssen CI-Prüfungen bestehen, die sowohl Linting als auch Formatierung umfassen
    - Behebe alle ESLint-Warnungen oder -Fehler vor dem Einreichen
    - Reagiere auf alle Rückmeldungen von Ellipsis, unserem automatisierten Code-Review-Tool
    - Folge TypeScript-Best-Practices und halte die Typsicherheit aufrecht

3. **Testen**

    - Füge Tests für neue Funktionen hinzu
    - Führe `npm test` aus, um sicherzustellen, dass alle Tests bestanden werden
    - Aktualisiere bestehende Tests, wenn deine Änderungen diese beeinflussen
    - Schließe sowohl Unit-Tests als auch Integrationstests ein, wo angemessen

4. **Commit-Richtlinien**

    - Schreibe klare, beschreibende Commit-Nachrichten
    - Verweise auf relevante Issues in Commits mit #issue-nummer

5. **Vor dem Einreichen**

    - Rebase deinen Branch auf den neuesten main-Branch
    - Stelle sicher, dass dein Branch erfolgreich baut
    - Überprüfe erneut, dass alle Tests bestanden werden
    - Prüfe deine Änderungen auf Debug-Code oder Konsolenausgaben

6. **Pull Request Beschreibung**
    - Beschreibe klar, was deine Änderungen bewirken
    - Füge Schritte zum Testen der Änderungen hinzu
    - Liste alle Breaking Changes auf
    - Füge Screenshots für UI-Änderungen hinzu

## Beitragsvereinbarung

Durch das Einreichen eines Pull Requests stimmst du zu, dass deine Beiträge unter derselben Lizenz wie das Projekt ([Apache 2.0](../LICENSE)) lizenziert werden.
